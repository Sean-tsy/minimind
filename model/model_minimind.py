import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig): # class 是类，类似图纸；这里定义了一个MiniMindConfig类，继承自PretrainedConfig，主要用于存储和管理模型的配置参数。
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs): # init是初始化函数，窗子实例时会自动调用的方法；python参数有位置参数、关键字参数、可变位置参数（tuple形式）和可变关键字参数（dict形式）
        super().__init__(**kwargs) # 用父类来实现初始化，**kwargs这里的用法是直接利用父类的__init__方法来处理那些MiniMindConfig没有显式声明但可能被传入的参数。
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0) # 将dropout参数从kwargs中提取出来，如果没有提供则默认为0.0；这种方式允许在创建MiniMindConfig实例时通过关键字参数传入dropout值，而不需要在MiniMindConfig的__init__方法中显式声明dropout参数。
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None # 如果inference_rope_scaling为True，则使用YaRN的rope_scaling配置，否则为None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps #是RMSNorm中的一个小常数，用于防止除以零的情况，确保数值稳定性；在计算标准差时，如果所有元素都相同，标准差将为零，这时加上eps可以避免除以零导致的错误。
        self.weight = nn.Parameter(torch.ones(dim)) #RMSNorm的权重参数，初始化为全1，训练过程中会更新；RMSNorm是一种归一化方法，类似于LayerNorm，但不减去均值，只除以标准差，因此计算更简单，性能更好。

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) #相比layernorm，rmsnorm没有减去均值的操作，因此计算更简单，性能更好

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):  #把位置编码“旋转进 Q 和 K 向量里”，提前计算好每个位置要用的 cos / sin
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0 #    先只取偶数维度的频率，再除以dim把维度映射到0～1之间，在用rope_base的幂函数得到每个维度对应的频率，取倒数是因为频率越高，位置编码变化越快，[: (dim // 2)]加入是防止越界，
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp 这里是对RoPE分段线性缩放，主要实现了根据位置的不同对频率进行不同程度的缩放，以增强模型在处理长序列时的表现；通过调整beta_fast、beta_slow和factor等参数，可以控制不同位置的频率缩放程度，从而提升模型在生成文本时的质量和连贯性。attn_factor是一个全局的缩放因子，用于调整注意力权重的缩放程度；通过这个参数可以进一步控制模型在处理长序列时的表现，增强模型对不同位置的关注度，从而提升生成文本的质量。
            beta_fast, beta_slow = rope_scaling["beta_fast"], rope_scaling["beta_slow"] #beta_fast和beta_slow分别是RoPE分段线性缩放的两个参数，控制了不同位置的频率缩放程度；beta_fast对应较快变化的频率，beta_slow对应较慢变化的频率，通过这两个参数可以实现对不同位置的频率进行不同程度的缩放，从而增强模型在处理长序列时的表现。
            factor, orig_max = rope_scaling["factor"], rope_scaling["original_max_position_embeddings"] #factor是RoPE分段线性缩放的缩放因子，orig_max是原始位置编码的最大位置，这两个参数一起决定了RoPE分段线性缩放的具体实现方式；factor控制了频率的缩放程度，orig_max则用于计算不同位置的频率缩放比例，从而实现对不同位置的频率进行不同程度的缩放。
            attn_factor = rope_scaling.get("attention_factor", 1.0) #attn_factor是RoPE分段线性缩放的注意力缩放因子，用于调整注意力权重的缩放程度；通过这个参数可以进一步控制模型在处理长序列时的表现，增强模型对不同位置的关注度，从而提升生成文本的质量。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base)) #inv_dim是一个函数，用于计算给定beta值对应的频率维度索引；通过这个函数可以根据beta_fast和beta_slow的值计算出对应的频率维度索引，从而实现对不同位置的频率进行不同程度的缩放。
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1) #low和high分别是RoPE分段线性缩放的频率维度索引范围，控制了哪些频率维度会被缩放；通过计算beta_fast和beta_slow对应的频率维度索引，可以确定哪些频率维度会被缩放，从而实现对不同位置的频率进行不同程度的缩放。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1) #ramp是一个线性递增的张量，用于实现RoPE分段线性缩放的过渡效果；通过这个ramp张量，可以在low和high之间实现频率的平滑过渡，从而增强模型在处理长序列时的表现，提升生成文本的质量和连贯性。
            freqs = freqs * (1 - ramp + ramp / factor) #通过对频率进行分段线性缩放，可以实现对不同位置的频率进行不同程度的缩放，从而增强模型在处理长序列时的表现；具体来说，freqs被乘以一个由ramp控制的缩放因子，这个缩放因子在low和high之间平滑过渡，从而实现对不同位置的频率进行不同程度的缩放。
    t = torch.arange(end, device=freqs.device) #t是一个从0到end-1的整数序列，用于表示位置索引；end参数决定了位置索引的范围，通常设置为模型能够处理的最大序列长度。
    freqs = torch.outer(t, freqs).float()#通过计算位置索引t与频率freqs的外积，可以得到一个形状为(end, dim//2)的张量，其中每个元素表示对应位置和频率的乘积。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor #希望能在attention计算中体现距离，用旋转来表示，所以取了两份cosine，sinusoidal的位置编码，并且乘以了attn_factor这个全局缩放因子，以调整注意力权重的缩放程度；通过这种方式，可以增强模型在处理长序列时的表现，提升生成文本的质量和连贯性。
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor 
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)#将最后一列一分为二，并将前半部分和后半部分交换位置（先写的后半部分），后半部分取负；这种旋转操作是RoPE位置编码的核心，通过这种方式将位置信息编码到查询和键的向量中，使得模型能够更好地捕捉序列中元素之间的相对位置信息，从而提升模型在处理长序列时的表现。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x #如果n_rep等于1，直接返回输入张量x，不进行任何操作；这是一个优化步骤，避免在不需要重复的情况下进行不必要的计算和内存使用。
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)) # 先用None增加一个维度，然后用expand广播复制，并没有真正复制数据而是变成n-rep个逻辑副本，最后reshape合并维度

class Attention(nn.Module): #定义了Attention类，采用GQA机制，其中查询有更多头数，而键值对共享较少的头数，这样可以在保持计算效率的同时提升模型的表达能力；GQA机制通过增加查询头数来增强模型的表示能力，同时通过共享键值头数来控制计算成本，使得模型在处理长序列时更高效。
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape #初始的时候，输入张量x的形状是(batch_size, seq_len, hidden_dim)，其中batch_size是批次大小，seq_len是序列长度，hidden_dim是隐藏层维度；通过x.shape获取这些维度信息，并将它们分别赋值给bsz、seq_len和_（下划线表示这个变量暂时不需要使用）。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x) #通过线性变换将输入张量x映射到查询、键和值的空间中，得到xq、xk和xv三个张量；这些张量的形状分别是(batch_size, seq_len, num_attention_heads * head_dim)和(batch_size, seq_len, num_key_value_heads * head_dim)，其中num_attention_heads是查询的头数，num_key_value_heads是键值对的头数，head_dim是每个头的维度。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  #用view切分查询张量xq，使其形状变为(batch_size, seq_len, n_local_heads, head_dim)，其中n_local_heads是查询的头数；同样地，使用view切分键和值张量xk和xv，使它们的形状变为(batch_size, seq_len, n_local_kv_heads, head_dim)，其中n_local_kv_heads是键值对的头数；这种切分方式为后续的注意力计算做好准备，使得每个头可以独立地处理自己的查询、键和值。
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings #来自Mini
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin) #通过apply_rotary_pos_emb将
        if past_key_value is not None: #如果past_key_value不为None，说明这是一个增量推理的步骤，此时需要将当前计算得到的键和值与之前缓存的键值进行拼接，以便在后续的注意力计算中使用完整的历史信息；这种机制允许模型在生成文本时逐步构建上下文，而不需要每次都重新计算整个序列的键值对，从而提高效率。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2)) # 复制KV张量来匹配查询头数，并将查询、键和值的维度从(batch_size, seq_len, num_heads, head_dim)转换为(batch_size, num_heads, seq_len, head_dim)，以便进行后续的注意力计算；这种转换使得每个头可以独立地处理自己的查询、键和值，同时保持了批次大小和序列长度的信息。
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)): 
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal) #当满足使用Flash Attention的条件时使用，主要是边计算部分QK softmax后边累加结果，提高计算速度；如果不满足条件，则使用传统的点积注意力计算方式。
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) #先交换键张量xk的最后两个维度，使其形状变为(batch_size, num_heads, head_dim, seq_len)（表示有seq_len个token，每个的维度是head_dim，前面两位不参与矩阵运算，只表示有多少次乘法，满足相等或有一个1即可（广播规则）），然后与查询张量xq进行矩阵乘法，得到注意力分数scores，其形状为(batch_size, num_heads, seq_len, seq_len)；最后将分数除以sqrt(head_dim)进行缩放，以保持数值稳定性。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1) #如果是因果注意力，用casual mask，只对倒数第seq_len个位置取到最后这部分加mask；构造mask时使用了torch.full创建一个全是负无穷的矩阵，并通过triu(1)方法将其转换为上三角矩阵（不包括对角线），这样就可以在计算注意力分数时有效地屏蔽掉未来位置的信息，确保模型只能关注当前和之前的位置，从而实现因果关系。上三角表示Query位置（行）只能看到Key位置（列）j满足j<=i的情况，保证了生成文本时的自回归性质。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9 #如果提供了attention_mask，则将其应用于注意力分数；通过unsqueeze增加维度（插入一个长度为1的维度，数字表示位置），使得attention_mask的形状与scores兼容，然后将mask中的0位置转换为一个很大的负数（-1e9），这样在softmax计算时这些位置的权重将接近于零，从而有效地屏蔽掉被mask的位置，确保模型不会关注这些位置的信息。
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            self._attn_weights = scores.detach() #保存注意力权重以供后续分析或可视化使用；通过detach()方法将其从计算图中分离出来，确保在反向传播过程中不会对这些权重进行梯度计算，从而节省内存和计算资源。
            output = self.attn_dropout(scores) @ xv  #先对注意力权重scores应用dropout，然后与值张量xv进行矩阵乘法，得到注意力输出output，其形状为(batch_size, num_heads, seq_len, head_dim)；这种计算方式实现了加权求和的操作，其中每个位置的输出是根据对应位置的查询与所有键之间的相似度（即注意力权重）对值进行加权求和得到的。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv 

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size #如果在初始化FeedForward实例时没有提供intermediate_size参数，则使用config.intermediate_size作为默认值；这种设计允许在创建FeedForward实例时通过关键字参数传入intermediate_size值，而不需要在FeedForward的__init__方法中显式声明intermediate_size参数。
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False) #把输入升维到 intermediate_size，其输出经过激活后当作门控信号。用来供激活函数判断
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False) #把门控后的结果从 intermediate_size 降回 hidden_size，接回残差流。
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False) #同样升维，但不经过激活，作为被门控的"内容信号"。由激活函数控制实际能通过多少
        self.act_fn = ACT2FN[config.hidden_act] #选择激活函数SiLU（也称为Swish），门控机制的直觉：up_proj(x) 是"想往下传的内容"，gate_proj(x) 是"允许多少内容通过的门"，act_fn是门的开关，down_proj是把最终结果映射回hidden_size以便与残差连接；通过这种设计，模型可以动态地控制每个位置的信息流动，从而提升模型的表达能力和性能。

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) #前向传播时，先通过gate_proj计算门控值，再通过act_fn激活函数处理，最后与up_proj的输出相乘，再通过down_proj映射回隐藏维度。

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)#
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]) 
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values): #把block和对应的past_key_value打包在一起，依次传入每个block进行前向传播；这种设计允许模型在生成文本时逐步构建上下文，而不需要每次都重新计算整个序列的键值对，从而提高效率。
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin): #定义了MiniMindForCausalLM类，继承自PreTrainedModel和GenerationMixin，主要用于实现基于MiniMind模型的语言生成任务；通过这种设计，模型可以在生成文本时根据当前的上下文信息预测下一个词，从而实现文本生成的功能。
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) #语言模型的输出层，将隐藏状态映射到词汇表大小的维度，以生成每个位置的下一个词的概率分布；通过这种设计，模型可以在生成文本时根据当前的上下文信息预测下一个词，从而实现文本生成的功能。
        self.model.embed_tokens.weight = self.lm_head.weight
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep 
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature #只取最后一个token的logits进行采样，temperature参数用于控制生成文本的多样性，较高的temperature会使得生成的文本更加多样化，而较低的temperature则会使得生成的文本更加集中和确定；通过这种方式，可以根据需要调整生成文本的质量和多样性。
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0: #top_p采样是一种基于累积概率的采样方法，通过计算每个token的概率分布，并根据top_p参数选择累积概率超过top_p的token作为候选集合，然后从这个集合中随机采样一个token作为下一个token；通过这种方式，可以根据需要调整生成文本的质量和多样性。
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)#softmax将logits转换为概率分布，multinomial根据这个分布随机采样一个token作为下一个token；如果do_sample为False，则直接选择概率最高的token作为下一个token；通过这种方式，可以根据需要调整生成文本的质量和多样性。
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids #返回生成的文本的token ID序列，如果return_kv为True，则返回一个包含生成的token ID序列和最终的past_key_values的字典；通过这种设计，用户可以根据需要选择是否获取生成文本的token ID序列以及对应的past_key_values，从而满足不同的使用场景和需求。