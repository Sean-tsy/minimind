import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import Siglip2ImageProcessor, Siglip2VisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)

class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=256, target_tokens=64):
        super().__init__()
        self.target_tokens = target_tokens
        self.merge = source_tokens // target_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * self.merge, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        b, n, d = x.shape
        x = x.reshape(b, self.target_tokens, d * self.merge)
        return self.mlp(x) #将输入的视觉特征张量进行重塑和线性变换，以适应语言模型的隐藏状态的维度；通过这种设计，可以将视觉特征有效地融入到语言模型的隐藏状态中，从而增强模型对视觉信息的理解和利用能力。

# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p16-ve"):
        self.config = config or VLMConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)

    @staticmethod #装饰器包装成静态方法，表示该方法不依赖于类的实例，可以直接通过类名调用；通过这种设计，可以在不创建类实例的情况下获取视觉模型和处理器，从而提高代码的灵活性和可重用性。
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()#设置transformers库的日志级别为错误，避免在加载视觉模型时输出过多的日志信息；通过这种方式，可以保持代码的清洁和专注于核心功能。
        if not os.path.exists(model_path):
            return None, None
        model = Siglip2VisionModel.from_pretrained(model_path) #从指定路径加载预训练的Siglip2视觉模型，如果路径不存在，则返回None；通过这种设计，可以根据需要选择是否使用视觉模型，从而满足不同的使用场景和需求。
        processor = Siglip2ImageProcessor.from_pretrained(model_path)#从指定路径加载预训练的Siglip2图像处理器，如果路径不存在，则返回None；通过这种设计，可以根据需要选择是否使用视觉处理器，从而满足不同的使用场景和需求。
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs #将输入图像转换为张量格式，使用预训练的图像处理器对图像进行处理，并返回一个包含处理后图像数据的字典；通过这种设计，可以方便地将输入图像转换为模型可接受的格式，从而实现视觉信息的处理和利用。

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()} #如果输入的图像数据是一个字典，并且其中的某些张量具有多余的维度（即第二维为1），则将这些张量的第二维压缩掉；通过这种设计，可以确保输入的图像数据具有正确的形状，从而避免在后续处理过程中出现维度不匹配的问题。
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state #返回视觉模型的最后隐藏状态，作为图像的特征表示；通过这种设计，可以将图像特征与语言模型的隐藏状态进行融合，从而实现多模态信息的处理和利用。

    @torch.compiler.disable#禁用PyTorch的编译器优化，确保该方法在执行时不会被编译器优化掉；通过这种设计，可以避免在处理视觉信息时出现潜在的性能问题，从而保证模型的稳定性和可靠性。
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512): # tokens: 输入的token ID序列，h: 语言模型的隐藏状态，vision_tensors: 视觉特征张量，seqlen: 序列长度；该方法的主要功能是将视觉特征张量插入到语言模型的隐藏状态中，以实现多模态信息的融合；通过这种设计，可以根据输入的token ID序列中的特殊标记位置，将对应的视觉特征插入到隐藏状态中，从而增强模型对视觉信息的理解和利用能力。
        if vision_tensors is None or not self.config.image_ids: #如果没有提供视觉特征张量或者配置中没有指定图像标记ID，则直接返回语言模型的隐藏状态；通过这种设计，可以在不使用视觉信息的情况下保持模型的正常运行，从而满足不同的使用场景和需求。
            return h
        marker, vf = self.config.image_ids[0], vision_tensors #获取图像标记ID和视觉特征张量；通过这种设计，可以根据配置中的图像标记ID来识别输入序列中的特殊标记位置，从而将对应的视觉特征插入到隐藏状态中，实现多模态信息的融合。
        if vf.dim() == 3:
            vf = vf.unsqueeze(1) # 如果视觉特征张量是三维的（即没有批次维度），则在第二维添加一个新的维度，以适应后续的处理；通过这种设计，可以确保视觉特征张量具有正确的形状，从而避免在后续处理过程中出现维度不匹配的问题。
        out = []
        for b in range(h.size(0)): #遍历batch
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0 #hb是当前批次样本的隐藏状态，seq是当前样本的token ID序列，k是视觉特征张量的索引，i是当前处理的位置索引；通过这种设计，可以逐步处理输入序列中的每个token，并根据特殊标记位置将对应的视觉特征插入到隐藏状态中，从而实现多模态信息的融合。
            while i < len(seq): 
                if seq[i] == marker: #token是图像站位符
                    start = i
                    while i < len(seq) and seq[i] == marker:#找连续的图像站位符
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen] #用图像的特征向量替换连续的图像站位符，并确保最终的隐藏状态长度不超过指定的序列长度；通过这种设计，可以将视觉特征有效地融入到语言模型的隐藏状态中，从而增强模型对视觉信息的理解和利用能力。
                        k += 1 
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out) #将处理后的隐藏状态列表堆叠成一个新的张量，并返回；通过这种设计，可以将处理后的隐藏状态以张量的形式返回，从而方便后续的处理和生成。

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids)) #将输入的token ID序列通过嵌入层转换为隐藏状态，并应用dropout进行正则化；通过这种设计，可以将输入的token ID序列转换为模型可处理的隐藏状态表示，从而为后续的处理和生成提供基础。

        if pixel_values is not None and start_pos == 0: #如果提供了图像数据，并且当前处理的位置是序列的起始位置，则将视觉特征张量插入到语言模型的隐藏状态中；通过这种设计，可以确保在处理输入序列的起始位置时，视觉信息能够被有效地融入到模型的隐藏状态中，从而增强模型对视觉信息的理解和利用能力。
            if hasattr(pixel_values, 'keys'):#支持多模态输入，如果输入的是字典形式的图像
                sample_val = next(iter(pixel_values.values())) #获取图像数据字典中的第一个张量，以检查其维度；通过这种设计，可以根据输入的图像数据的形状来确定批次大小和图像数量，从而正确地处理视觉信息。
                if sample_val.ndim == 5: #如果图像数据是五维的（即包含批次维度、图像数量维度、通道维度、高度维度和宽度维度），则将其重塑为四维的（即批次大小、图像数量、通道数、特征长度），并通过视觉投影层进行处理。
                    bs, num = sample_val.shape[:2] 
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings({k: v.flatten(0, 1) for k, v in pixel_values.items()}, self.vision_encoder)).view(bs, num, self.config.image_token_len, -1)#现将b和n合并然后提取视觉特征，再通过投影层处理，最后再reshape回来
                else:
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder))
            else:#输入是普通的张量形式的图像数据
                if len(pixel_values.shape) == 6: #processor处理后可能会在第二维添加一个图像数量维度，如果存在则将其压缩掉；通过这种设计，可以确保输入的图像数据具有正确的形状，从而避免在后续处理过程中出现维度不匹配的问题。
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape#获取维度
                vision_tensors = torch.stack([self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)) for i in range(num)], dim=1)#另一种处理方式应对输入格式不统一，一张一张地处理后再堆叠起来，形状维持为bs，n，token_len，hidden_size
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=input_ids.shape[1]) #形状变成了bs，seq,c，视觉特征被插入到语言模型的隐藏状态中；通过这种设计，可以将视觉特征有效地融入到语言模型的隐藏状态中，从而增强模型对视觉信息的理解和利用能力。

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)