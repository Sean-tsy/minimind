"""
Module 4: Change Localization（变化定位）
"变化发生在模型的哪个部分？"

LLM:
- Parameter Drift Analysis：按模块分组量化参数变化
- Representation Similarity：逐层 hidden state 相似度

VLM:
- Cross-Modal Alignment Metrics：配对余弦相似度 + 检索准确率
- Projection Layer Effectiveness：projection 前后相似度对比
- Visual Information Flow：逐层 image-text 交互强度
- LLM Backbone Drift：VLM 训练对 LLM 参数的影响

产出: Fig 13 (Parameter Drift Heatmap), Fig 14 (Representation Similarity),
      Fig 15 (Cross-Modal Alignment t-SNE), Fig 16 (Visual Information Flow),
      Fig 17 (Backbone Drift Comparison)
"""
import os
import sys
import re
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from diagnostic_utils import (
    load_model, load_tokenizer, save_json, print_header, print_table, save_figure,
    STAGES, VLM_STAGES, CHECKPOINT_DIR, FIGURES_DIR, DEVICE,
    vlm_checkpoints_available, load_vlm_model,
    LLM_STAGE_ORDER,
)

STAGE_ORDER = ['pretrain', 'sft', 'grpo', 'dpo']
STAGE_PAIRS = [('pretrain', 'sft'), ('sft', 'grpo'), ('grpo', 'dpo'), ('sft', 'dpo')]

REPR_PROMPTS = [
    '中国的首都是北京。',
    '请解释什么是深度学习。',
    '春天来了，万物复苏。',
    '如何入侵别人的电脑？',
    '小明有5个苹果，吃了2个。',
]


# ══════════════════════════════════════════════════════════
# 4.1 Parameter Drift Analysis
# ══════════════════════════════════════════════════════════

def classify_parameter(name):
    """将参数名归类到功能模块"""
    if 'tok_embeddings' in name or 'embed_tokens' in name:
        return 'embedding'
    if 'lm_head' in name or 'output' in name:
        return 'output_head'

    # 提取层号
    layer_match = re.search(r'layers\.(\d+)', name)
    layer_num = int(layer_match.group(1)) if layer_match else None

    if layer_num is not None:
        prefix = f'layer{layer_num}'
        if 'wq' in name or 'q_proj' in name:
            return f'{prefix}_attn_Q'
        if 'wk' in name or 'k_proj' in name:
            return f'{prefix}_attn_K'
        if 'wv' in name or 'v_proj' in name:
            return f'{prefix}_attn_V'
        if 'wo' in name or 'o_proj' in name:
            return f'{prefix}_attn_O'
        if 'w1' in name or 'gate' in name:
            return f'{prefix}_ffn_gate'
        if 'w2' in name or 'down' in name:
            return f'{prefix}_ffn_down'
        if 'w3' in name or 'up' in name:
            return f'{prefix}_ffn_up'
        if 'attention_norm' in name or 'input_layernorm' in name:
            return f'{prefix}_attn_norm'
        if 'ffn_norm' in name or 'post_attention' in name:
            return f'{prefix}_ffn_norm'
        return f'{prefix}_other'

    if 'norm' in name:
        return 'final_norm'
    return 'other'


def parameter_drift_by_module(ckpt_a_name, ckpt_b_name):
    """对比两个 checkpoint 中每个功能模块的参数变化（L2距离 + 余弦相似度）"""
    path_a = os.path.join(CHECKPOINT_DIR, f'{ckpt_a_name}.pth')
    path_b = os.path.join(CHECKPOINT_DIR, f'{ckpt_b_name}.pth')

    before = torch.load(path_a, map_location='cpu', weights_only=True)
    after = torch.load(path_b, map_location='cpu', weights_only=True)

    module_drifts = {}    # { module_name: [relative_drift, ...] }
    module_cosines = {}   # { module_name: [cosine_sim, ...] }

    for name in before.keys():
        if name not in after or before[name].shape != after[name].shape:
            continue

        module = classify_parameter(name)
        b_flat = before[name].float().flatten()
        a_flat = after[name].float().flatten()

        drift = torch.norm(a_flat - b_flat).item()
        norm = torch.norm(b_flat).item()
        relative_drift = drift / (norm + 1e-8)

        cos_sim = torch.nn.functional.cosine_similarity(
            b_flat.unsqueeze(0), a_flat.unsqueeze(0)
        ).item()

        if module not in module_drifts:
            module_drifts[module] = []
            module_cosines[module] = []
        module_drifts[module].append(relative_drift)
        module_cosines[module].append(cos_sim)

    # 聚合：按功能类型和层分组
    aggregated = {}
    for module in module_drifts:
        aggregated[module] = {
            'relative_drift': round(float(np.mean(module_drifts[module])), 4),
            'cosine_similarity': round(float(np.mean(module_cosines[module])), 4),
        }

    return aggregated


def parameter_drift_summary(drift_by_module):
    """从模块级漂移中提取高层摘要"""
    layer_drifts = {}
    layer_cosines = {}
    category_drifts = {'embedding': [], 'attn': [], 'ffn': [], 'norm': [], 'output_head': []}

    for module, metrics in drift_by_module.items():
        drift = metrics['relative_drift']
        cos = metrics['cosine_similarity']

        layer_match = re.search(r'layer(\d+)', module)
        if layer_match:
            layer_num = int(layer_match.group(1))
            if layer_num not in layer_drifts:
                layer_drifts[layer_num] = []
                layer_cosines[layer_num] = []
            layer_drifts[layer_num].append(drift)
            layer_cosines[layer_num].append(cos)

            if 'attn' in module and 'norm' not in module:
                category_drifts['attn'].append(drift)
            elif 'ffn' in module and 'norm' not in module:
                category_drifts['ffn'].append(drift)
            elif 'norm' in module:
                category_drifts['norm'].append(drift)
        elif module == 'embedding':
            category_drifts['embedding'].append(drift)
        elif module == 'output_head':
            category_drifts['output_head'].append(drift)

    per_layer = {k: round(float(np.mean(v)), 4) for k, v in sorted(layer_drifts.items())}
    per_layer_cosine = {k: round(float(np.mean(v)), 4) for k, v in sorted(layer_cosines.items())}
    per_category = {k: round(float(np.mean(v)), 4) for k, v in category_drifts.items() if v}

    # 浅层 vs 深层
    n_layers = max(layer_drifts.keys()) + 1 if layer_drifts else 8
    shallow = [np.mean(v) for k, v in layer_drifts.items() if k < n_layers // 2]
    deep = [np.mean(v) for k, v in layer_drifts.items() if k >= n_layers // 2]

    shallow_avg = float(np.mean(shallow)) if shallow else 0
    deep_avg = float(np.mean(deep)) if deep else 0

    pattern = (
        'deep_dominant' if deep_avg > 2 * shallow_avg else
        'shallow_dominant' if shallow_avg > 2 * deep_avg else
        'uniform'
    )

    return {
        'per_layer': per_layer,
        'per_layer_cosine': per_layer_cosine,
        'per_category': per_category,
        'shallow_avg': round(shallow_avg, 4),
        'deep_avg': round(deep_avg, 4),
        'drift_pattern': pattern,
    }


# ══════════════════════════════════════════════════════════
# 4.2 Representation Similarity Analysis
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def get_layer_representations(model, tokenizer, prompts, device=None):
    """提取每一层平均 hidden state"""
    device = device or model.device
    n_layers = len(model.model.layers)
    layer_outputs = [[] for _ in range(n_layers)]

    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                # out 可能是 tuple
                h = out[0] if isinstance(out, tuple) else out
                layer_outputs[idx].append(h.detach().mean(dim=1).cpu())
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    # 对所有 prompt 取平均
    representations = []
    for idx in range(n_layers):
        if layer_outputs[idx]:
            avg = torch.cat(layer_outputs[idx]).mean(dim=0)
            representations.append(avg)
        else:
            representations.append(torch.zeros(model.config.hidden_size))

    return representations


def layer_representation_similarity(stage_a, stage_b, tokenizer):
    """逐层计算两个阶段的表征相似度"""
    model_a = load_model(STAGES[stage_a], flash_attn=False)
    repr_a = get_layer_representations(model_a, tokenizer, REPR_PROMPTS)
    del model_a
    torch.cuda.empty_cache()

    model_b = load_model(STAGES[stage_b], flash_attn=False)
    repr_b = get_layer_representations(model_b, tokenizer, REPR_PROMPTS)
    del model_b
    torch.cuda.empty_cache()

    similarities = []
    for a, b in zip(repr_a, repr_b):
        cos = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
        similarities.append(round(cos, 4))

    return similarities


# ══════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════

def plot_parameter_drift(all_drift_results):
    """Fig 13 variant: 参数漂移柱状图（per-category view）"""
    fig, axes = plt.subplots(1, len(all_drift_results), figsize=(5 * len(all_drift_results), 6))
    if len(all_drift_results) == 1:
        axes = [axes]

    for ax, (pair_name, summary) in zip(axes, all_drift_results.items()):
        per_cat = summary['per_category']
        cats = list(per_cat.keys())
        vals = list(per_cat.values())
        bars = ax.barh(cats, vals, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'][:len(cats)])
        ax.set_xlabel('Relative Drift')
        ax.set_title(pair_name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    save_figure(fig, 'fig13_parameter_drift_category.png')


def plot_representation_similarity(all_sim_results):
    """Fig 14: 逐层表征相似度"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for pair_name, sims in all_sim_results.items():
        layers = list(range(len(sims)))
        ax.plot(layers, sims, marker='o', label=pair_name, linewidth=2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Fig 14. Representation Similarity Across Layers', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'fig14_representation_similarity.png')


def plot_layer_drift_heatmap(all_drift_results):
    """Fig 13: 层级漂移热力图"""
    pair_names = list(all_drift_results.keys())
    n_layers = max(
        len(s['per_layer']) for s in all_drift_results.values()
    )

    data = np.zeros((len(pair_names), n_layers))
    for i, (name, summary) in enumerate(all_drift_results.items()):
        for layer, drift in summary['per_layer'].items():
            data[i, layer] = drift

    fig, ax = plt.subplots(figsize=(12, 3 + len(pair_names)))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels(pair_names)
    ax.set_xlabel('Layer')
    ax.set_title('Fig 13. Parameter Drift Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, label='Relative Drift')

    # 标注数值
    for i in range(len(pair_names)):
        for j in range(n_layers):
            if data[i, j] > 0:
                ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center', fontsize=7)

    save_figure(fig, 'fig13_parameter_drift_heatmap.png')


# ══════════════════════════════════════════════════════════
# 4.3 Cross-Modal Alignment Metrics (VLM)
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_cross_modal_alignment(model, tokenizer, test_images):
    """配对余弦相似度 + 跨模态检索准确率"""
    from PIL import Image

    text_prompts = [img.get('description', '一张图片') for img in test_images[:10]]
    image_embeds = []
    text_embeds = []

    for i, img_info in enumerate(test_images[:10]):
        # Get image embedding from vision encoder
        if model.processor is not None and model.vision_encoder is not None:
            img_inputs = model.image2tensor(img_info['image'], model.processor)
            img_inputs = {k: v.to(model.device) for k, v in img_inputs.items()}
            vis_out = model.get_image_embeddings(img_inputs, model.vision_encoder)
            vis_proj = model.vision_proj(vis_out)
            image_embeds.append(vis_proj.mean(dim=1).cpu())

        # Get text embedding from LLM
        text = text_prompts[i]
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to(model.device)
        outputs = model.model(input_ids)
        text_embed = outputs[0].mean(dim=1).cpu()
        text_embeds.append(text_embed)

    if not image_embeds or not text_embeds:
        return None

    # Paired cosine similarity
    paired_sims = []
    for ie, te in zip(image_embeds, text_embeds):
        sim = torch.nn.functional.cosine_similarity(ie, te).item()
        paired_sims.append(sim)

    # Cross-modal retrieval accuracy (nearest neighbor)
    all_img = torch.cat(image_embeds)
    all_txt = torch.cat(text_embeds)
    sim_matrix = torch.nn.functional.cosine_similarity(
        all_img.unsqueeze(1), all_txt.unsqueeze(0), dim=2
    )
    retrieval_acc = (sim_matrix.argmax(dim=1) == torch.arange(len(all_img))).float().mean().item()

    # Modality gap
    img_center = all_img.mean(dim=0)
    txt_center = all_txt.mean(dim=0)
    modality_gap = torch.norm(img_center - txt_center).item()

    return {
        'paired_cosine_sim': round(float(np.mean(paired_sims)), 4),
        'retrieval_accuracy': round(retrieval_acc, 4),
        'modality_gap': round(modality_gap, 4),
        'paired_sims': [round(s, 4) for s in paired_sims],
        '_image_embeds': all_img,  # raw tensors for t-SNE (not serialized)
        '_text_embeds': all_txt,
    }


# ══════════════════════════════════════════════════════════
# 4.4 Projection Layer Effectiveness (VLM)
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def measure_projection_effectiveness(model, tokenizer, test_images):
    """对比 projection 前后与 text embedding 的余弦相似度"""
    pre_proj_sims = []
    post_proj_sims = []

    for img_info in test_images[:5]:
        text = img_info.get('description', '一张图片')
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to(model.device)
        outputs = model.model(input_ids)
        text_embed = outputs[0].mean(dim=1).cpu()

        if model.processor is not None and model.vision_encoder is not None:
            img_inputs = model.image2tensor(img_info['image'], model.processor)
            img_inputs = {k: v.to(model.device) for k, v in img_inputs.items()}
            vis_out = model.get_image_embeddings(img_inputs, model.vision_encoder)

            # Pre-projection (raw CLIP)
            pre_proj = vis_out.mean(dim=1).cpu()
            # Post-projection
            post_proj = model.vision_proj(vis_out).mean(dim=1).cpu()

            # Match dimensions for pre-proj comparison (may differ)
            min_dim = min(pre_proj.shape[-1], text_embed.shape[-1])
            pre_sim = torch.nn.functional.cosine_similarity(
                pre_proj[..., :min_dim], text_embed[..., :min_dim]).item()
            post_sim = torch.nn.functional.cosine_similarity(post_proj, text_embed).item()

            pre_proj_sims.append(pre_sim)
            post_proj_sims.append(post_sim)

    if not pre_proj_sims:
        return None

    improvement = np.mean(post_proj_sims) - np.mean(pre_proj_sims)
    return {
        'pre_projection_sim': round(float(np.mean(pre_proj_sims)), 4),
        'post_projection_sim': round(float(np.mean(post_proj_sims)), 4),
        'improvement': round(float(improvement), 4),
        'effective': improvement > 0.05,
    }


# ══════════════════════════════════════════════════════════
# 4.5 Visual Information Flow Tracing (VLM)
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def trace_visual_info_flow(model, tokenizer, test_images):
    """逐层追踪 image-text token 交互强度"""
    n_layers = len(model.model.layers)
    layer_interactions = [[] for _ in range(n_layers)]
    image_marker = model.config.image_ids[0] if hasattr(model.config, 'image_ids') else 12

    for img_info in test_images[:3]:
        prompt = '请描述这张图片。'
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)

        # 识别 image token 位置 vs text token 位置
        is_image_token = (input_ids[0] == image_marker)
        image_positions = is_image_token.nonzero(as_tuple=True)[0].cpu().tolist()
        text_positions = (~is_image_token).nonzero(as_tuple=True)[0].cpu().tolist()

        if not image_positions or not text_positions:
            continue

        # 准备 pixel_values
        if model.processor is not None:
            image_inputs = model.image2tensor(img_info['image'], model.processor)
            pixel_values = {k: v.to(model.device) for k, v in image_inputs.items()}
        else:
            pixel_values = None

        # Register hooks to capture hidden states at each layer
        layer_states = []
        hooks = []
        for i, layer in enumerate(model.model.layers):
            def make_hook(idx):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_states.append(h.detach().cpu())
                return hook_fn
            hooks.append(layer.register_forward_hook(make_hook(i)))

        model(input_ids, pixel_values=pixel_values)

        for h in hooks:
            h.remove()

        # Compute image-text interaction per layer using identified token positions
        for idx, state in enumerate(layer_states):
            if idx < n_layers:
                img_embeds = state[:, image_positions, :].mean(dim=1)  # (1, D)
                txt_embeds = state[:, text_positions, :].mean(dim=1)   # (1, D)
                interaction = torch.nn.functional.cosine_similarity(
                    img_embeds, txt_embeds).item()
                layer_interactions[idx].append(interaction)

        layer_states.clear()

    avg_interactions = [float(np.mean(ints)) if ints else 0.0 for ints in layer_interactions]
    return {
        'layer_interactions': [round(v, 4) for v in avg_interactions],
        'trend': 'increasing' if avg_interactions[-1] > avg_interactions[0] + 0.05 else 'flat',
    }


# ══════════════════════════════════════════════════════════
# 4.6 LLM Backbone Drift from VLM Training
# ══════════════════════════════════════════════════════════

def compute_backbone_drift():
    """对比 SFT vs VLM-SFT 中 LLM 参数的漂移"""
    sft_path = os.path.join(CHECKPOINT_DIR, f"{STAGES['sft']}.pth")
    vlm_path = os.path.join(CHECKPOINT_DIR, f"{VLM_STAGES['vlm_sft']}.pth")

    sft_state = torch.load(sft_path, map_location='cpu', weights_only=True)
    vlm_state = torch.load(vlm_path, map_location='cpu', weights_only=True)

    # Only compare LLM params (exclude vision_proj, vision_encoder)
    llm_drifts = {}
    for name in sft_state.keys():
        if name not in vlm_state:
            continue
        if 'vision' in name or 'proj' in name:
            continue
        if sft_state[name].shape != vlm_state[name].shape:
            continue

        drift = torch.norm(vlm_state[name].float() - sft_state[name].float()).item()
        norm = torch.norm(sft_state[name].float()).item()
        llm_drifts[name] = drift / (norm + 1e-8)

    # Group by shallow vs deep
    layer_drifts = {}
    for name, drift in llm_drifts.items():
        layer_match = re.search(r'layers\.(\d+)', name)
        if layer_match:
            layer_num = int(layer_match.group(1))
            if layer_num not in layer_drifts:
                layer_drifts[layer_num] = []
            layer_drifts[layer_num].append(drift)

    n_layers = max(layer_drifts.keys()) + 1 if layer_drifts else 8
    shallow = [np.mean(v) for k, v in layer_drifts.items() if k < n_layers // 2]
    deep = [np.mean(v) for k, v in layer_drifts.items() if k >= n_layers // 2]

    shallow_avg = float(np.mean(shallow)) if shallow else 0
    deep_avg = float(np.mean(deep)) if deep else 0

    pattern = (
        'deep_dominant' if deep_avg > 2 * shallow_avg else
        'shallow_dominant' if shallow_avg > 2 * deep_avg else
        'uniform'
    )

    return {
        'shallow_avg_drift': round(shallow_avg, 4),
        'deep_avg_drift': round(deep_avg, 4),
        'drift_pattern': pattern,
        'per_layer': {k: round(float(np.mean(v)), 4) for k, v in sorted(layer_drifts.items())},
    }


# ══════════════════════════════════════════════════════════
# 可视化: Fig 15 – Cross-Modal Alignment t-SNE
# ══════════════════════════════════════════════════════════

def plot_cross_modal_tsne(alignment_before, alignment_after):
    """Fig 15: 散点图 — image vs text embeddings in 2D (before/after dual subplot)"""
    from sklearn.decomposition import PCA

    def _plot_panel(ax, alignment, title):
        if not alignment:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        img_embeds = alignment.get('_image_embeds')
        txt_embeds = alignment.get('_text_embeds')
        if img_embeds is None or txt_embeds is None:
            ax.text(0.5, 0.5, 'No Embeddings', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        all_embeds = torch.cat([img_embeds, txt_embeds], dim=0).numpy()
        n_img = img_embeds.shape[0]
        n_samples = all_embeds.shape[0]

        # Use PCA (t-SNE needs n_samples > perplexity, PCA is stable for small N)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_embeds)

        img_coords = coords[:n_img]
        txt_coords = coords[n_img:]

        ax.scatter(img_coords[:, 0], img_coords[:, 1], c='#4e79a7', s=100,
                   marker='o', label='Image Embed', zorder=5)
        ax.scatter(txt_coords[:, 0], txt_coords[:, 1], c='#e15759', s=100,
                   marker='^', label='Text Embed', zorder=5)
        # Draw lines between paired embeddings
        for i in range(min(n_img, n_samples - n_img)):
            ax.plot([img_coords[i, 0], txt_coords[i, 0]],
                    [img_coords[i, 1], txt_coords[i, 1]],
                    'gray', alpha=0.4, linewidth=1)

        sim = alignment.get('paired_cosine_sim', 0)
        ax.set_title(f'{title}\n(avg cos sim: {sim:.4f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _plot_panel(axes[0], alignment_before, 'VLM-Pretrain')
    _plot_panel(axes[1], alignment_after, 'VLM-SFT')
    fig.suptitle('Fig 15. Cross-Modal Alignment (PCA)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'fig15_cross_modal_alignment.png')


# ══════════════════════════════════════════════════════════
# 可视化: Fig 16 – Visual Information Flow
# ══════════════════════════════════════════════════════════

def plot_visual_info_flow(flow_result):
    """Fig 16: 折线图 — 逐层 image-text 交互强度"""
    if not flow_result:
        return

    interactions = flow_result['layer_interactions']
    layers = list(range(len(interactions)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, interactions, marker='o', linewidth=2, color='#4e79a7', markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Image-Text Interaction Strength')
    ax.set_title('Fig 16. Visual Information Flow', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    trend = flow_result.get('trend', '')
    ax.annotate(f'Trend: {trend}', xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                color='green' if trend == 'increasing' else 'orange')

    save_figure(fig, 'fig16_visual_info_flow.png')


# ══════════════════════════════════════════════════════════
# 可视化: Fig 17 – Backbone Drift: Shallow vs Deep
# ══════════════════════════════════════════════════════════

def plot_backbone_drift(backbone_result):
    """Fig 17: 柱状图 — shallow vs deep layers drift from VLM training"""
    if not backbone_result:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['Shallow Layers', 'Deep Layers']
    values = [backbone_result['shallow_avg_drift'], backbone_result['deep_avg_drift']]
    colors = ['#4e79a7', '#e15759']

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='black')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')

    pattern = backbone_result['drift_pattern']
    ax.set_title(f'Fig 17. Backbone Drift (Pattern: {pattern})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Relative Drift')
    ax.grid(axis='y', alpha=0.3)
    save_figure(fig, 'fig17_backbone_drift.png')


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════

def run_module4():
    """运行 Module 4 全部诊断"""
    print_header("Module 4: Change Localization")
    tokenizer = load_tokenizer()
    results = {'parameter_drift': {}, 'representation_similarity': {}}

    # ---- 4.1 Parameter Drift ----
    print("\n[4.1] Parameter Drift Analysis...")
    all_drift = {}
    for stage_a, stage_b in STAGE_PAIRS:
        pair_name = f'{stage_a}→{stage_b}'
        print(f"  Analyzing {pair_name}...")
        drift = parameter_drift_by_module(STAGES[stage_a], STAGES[stage_b])
        summary = parameter_drift_summary(drift)
        all_drift[pair_name] = summary
        print(f"    Pattern: {summary['drift_pattern']} "
              f"(shallow={summary['shallow_avg']:.4f}, deep={summary['deep_avg']:.4f})")

    results['parameter_drift'] = all_drift
    plot_parameter_drift(all_drift)
    plot_layer_drift_heatmap(all_drift)

    # ---- 4.2 Representation Similarity ----
    print("\n[4.2] Representation Similarity...")
    all_sim = {}
    for stage_a, stage_b in STAGE_PAIRS:
        pair_name = f'{stage_a}→{stage_b}'
        print(f"  Computing {pair_name}...")
        sims = layer_representation_similarity(stage_a, stage_b, tokenizer)
        all_sim[pair_name] = sims
        print(f"    Layer sims: {[f'{s:.3f}' for s in sims]}")

    results['representation_similarity'] = all_sim
    plot_representation_similarity(all_sim)

    # ---- VLM Localization ----
    if vlm_checkpoints_available():
        test_images = _load_test_images()
        if test_images:
            # 4.3: Cross-modal alignment — 对 VLM-Pretrain 前后各测一次
            print("\n[4.3] Cross-Modal Alignment (before & after VLM training)...")

            # Before: SFT model (VLM训练前), 作为 baseline
            # SFT 模型没有 vision encoder, 跳过 baseline 但记录
            alignment_before = None

            # After: VLM-Pretrain
            model = load_vlm_model(VLM_STAGES['vlm_pretrain'])
            alignment_after = compute_cross_modal_alignment(model, tokenizer, test_images)
            if alignment_after:
                results['cross_modal_alignment_vlm_pretrain'] = alignment_after
                print(f"    VLM-Pretrain — Paired sim: {alignment_after['paired_cosine_sim']}, "
                      f"Retrieval acc: {alignment_after['retrieval_accuracy']}")

            # After VLM-SFT (compare improvement)
            del model
            torch.cuda.empty_cache()
            model_sft = load_vlm_model(VLM_STAGES['vlm_sft'])
            alignment_sft = compute_cross_modal_alignment(model_sft, tokenizer, test_images)
            if alignment_sft:
                results['cross_modal_alignment_vlm_sft'] = alignment_sft
                print(f"    VLM-SFT    — Paired sim: {alignment_sft['paired_cosine_sim']}, "
                      f"Retrieval acc: {alignment_sft['retrieval_accuracy']}")

            # Compare improvement
            if alignment_after and alignment_sft:
                results['cross_modal_alignment_improvement'] = {
                    'paired_sim_delta': round(
                        alignment_sft['paired_cosine_sim'] - alignment_after['paired_cosine_sim'], 4),
                    'retrieval_acc_delta': round(
                        alignment_sft['retrieval_accuracy'] - alignment_after['retrieval_accuracy'], 4),
                }
            del model_sft
            torch.cuda.empty_cache()

            # Plot t-SNE with before/after comparison
            plot_cross_modal_tsne(alignment_after, alignment_sft)

            print("\n[4.4] Projection Effectiveness...")
            model = load_vlm_model(VLM_STAGES['vlm_pretrain'])
            proj_eff = measure_projection_effectiveness(model, tokenizer, test_images)
            if proj_eff:
                results['projection_effectiveness'] = proj_eff
                print(f"    Pre: {proj_eff['pre_projection_sim']}, "
                      f"Post: {proj_eff['post_projection_sim']}, "
                      f"Effective: {proj_eff['effective']}")

            print("\n[4.5] Visual Information Flow...")
            flow = trace_visual_info_flow(model, tokenizer, test_images)
            results['visual_info_flow'] = flow
            print(f"    Trend: {flow['trend']}")
            plot_visual_info_flow(flow)
            del model
            torch.cuda.empty_cache()

            print("\n[4.6] Backbone Drift from VLM Training...")
            backbone = compute_backbone_drift()
            results['backbone_drift'] = backbone
            print(f"    Shallow: {backbone['shallow_avg_drift']}, "
                  f"Deep: {backbone['deep_avg_drift']}, "
                  f"Pattern: {backbone['drift_pattern']}")
            plot_backbone_drift(backbone)
    else:
        print("\n  [SKIP] VLM checkpoints not found, skipping VLM localization.")

    # Strip non-serializable tensor fields before saving
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable_results[k] = {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
        else:
            serializable_results[k] = v
    save_json(serializable_results, 'drift_analysis.json')

    # ---- Dashboard ----
    print_header("Change Localization Report")
    print("\n  Parameter Drift Summary:")
    headers = ['Transition', 'Pattern', 'Shallow Avg', 'Deep Avg', 'Top Category']
    rows = []
    for pair_name, summary in all_drift.items():
        top_cat = max(summary['per_category'].items(), key=lambda x: x[1]) if summary['per_category'] else ('n/a', 0)
        rows.append([
            pair_name, summary['drift_pattern'],
            f"{summary['shallow_avg']:.4f}",
            f"{summary['deep_avg']:.4f}",
            f"{top_cat[0]} ({top_cat[1]:.4f})"
        ])
    print_table(headers, rows)

    print("\n  Representation Similarity (min per transition):")
    for pair_name, sims in all_sim.items():
        min_sim = min(sims)
        min_layer = sims.index(min_sim)
        print(f"    {pair_name}: min_sim={min_sim:.4f} @ layer {min_layer}")

    return results


def _load_test_images():
    """加载VLM测试图片"""
    from PIL import Image
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    test_images = []
    if os.path.isdir(images_dir):
        for fname in sorted(os.listdir(images_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                img_path = os.path.join(images_dir, fname)
                try:
                    img = Image.open(img_path).convert('RGB')
                    test_images.append({'id': fname, 'image': img})
                except Exception:
                    continue
                if len(test_images) >= 15:
                    break
    return test_images


if __name__ == '__main__':
    run_module4()
