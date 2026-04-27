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
    load_test_images,
    STAGES, VLM_STAGES, CHECKPOINT_DIR, FIGURES_DIR, DEVICE,
    vlm_checkpoints_available, load_vlm_model,
    LLM_STAGE_ORDER,
)
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
    # v6: per-layer per-category for Attn×FFN × Shallow×Deep grouping
    attn_per_layer = {}
    ffn_per_layer = {}

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
                attn_per_layer.setdefault(layer_num, []).append(drift)
            elif 'ffn' in module and 'norm' not in module:
                category_drifts['ffn'].append(drift)
                ffn_per_layer.setdefault(layer_num, []).append(drift)
            elif 'norm' in module:
                category_drifts['norm'].append(drift)
        elif module == 'embedding':
            category_drifts['embedding'].append(drift)
        elif module == 'output_head':
            category_drifts['output_head'].append(drift)

    per_layer = {k: round(float(np.mean(v)), 4) for k, v in sorted(layer_drifts.items())}
    per_layer_cosine = {k: round(float(np.mean(v)), 4) for k, v in sorted(layer_cosines.items())}
    per_category = {k: round(float(np.mean(v)), 4) for k, v in category_drifts.items() if v}
    attn_by_layer = {k: round(float(np.mean(v)), 4) for k, v in sorted(attn_per_layer.items())}
    ffn_by_layer = {k: round(float(np.mean(v)), 4) for k, v in sorted(ffn_per_layer.items())}

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
        'attn_per_layer': attn_by_layer,
        'ffn_per_layer': ffn_by_layer,
        'shallow_avg': round(shallow_avg, 4),
        'deep_avg': round(deep_avg, 4),
        'drift_pattern': pattern,
    }


# ══════════════════════════════════════════════════════════
# 4.2 Representation Similarity Analysis
# ══════════════════════════════════════════════════════════

def linear_cka(X, Y, eps=1e-8):
    """Linear CKA (Kornblith et al., 2019).

    X, Y: [N, D] tensors (N = #samples, D = hidden dim).
    返回 [0, 1] 标量。对正交变换、各向同性缩放不变。
    采用高效形式: CKA = ||X^T Y||_F^2 / (||X^T X||_F · ||Y^T Y||_F),
    其中 X, Y 已做样本维零均值化。
    """
    X = X.float()
    Y = Y.float()
    # center along sample dim
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # cross-covariance Frobenius norm squared
    xty = X.t() @ Y                       # [D, D]
    num = (xty ** 2).sum()                # ||X^T Y||_F^2

    xtx = X.t() @ X
    yty = Y.t() @ Y
    denom = torch.sqrt((xtx ** 2).sum()) * torch.sqrt((yty ** 2).sum())
    return float((num / (denom + eps)).item())


@torch.no_grad()
def get_layer_representations(model, tokenizer, prompts, device=None):
    """提取每一层 hidden state, 返回 list of [N, D] 矩阵 (N=#prompts)."""
    device = device or model.device
    n_layers = len(model.model.layers)
    layer_outputs = [[] for _ in range(n_layers)]

    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                # out 可能是 tuple; 对 seq_len 求平均 -> [1, D]
                h = out[0] if isinstance(out, tuple) else out
                layer_outputs[idx].append(h.detach().mean(dim=1).cpu())
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    # 保留 prompt 维: 每层得到 [N_prompts, D] 矩阵
    representations = []
    for idx in range(n_layers):
        if layer_outputs[idx]:
            mat = torch.cat(layer_outputs[idx], dim=0)  # [N, D]
            representations.append(mat)
        else:
            representations.append(torch.zeros(len(prompts), model.config.hidden_size))

    return representations


def layer_representation_similarity(stage_a, stage_b, tokenizer):
    """逐层计算两个阶段的表征相似度: 同时返回 Linear CKA 与平均 cosine.

    返回: {'cka': [...per layer], 'cosine': [...per layer]}
    """
    model_a = load_model(STAGES[stage_a], flash_attn=False)
    repr_a = get_layer_representations(model_a, tokenizer, REPR_PROMPTS)
    del model_a
    torch.cuda.empty_cache()

    model_b = load_model(STAGES[stage_b], flash_attn=False)
    repr_b = get_layer_representations(model_b, tokenizer, REPR_PROMPTS)
    del model_b
    torch.cuda.empty_cache()

    cka_list, cos_list = [], []
    for A, B in zip(repr_a, repr_b):
        # CKA: 正交/缩放不变, 在 [N, D] 上比较结构
        cka_list.append(round(linear_cka(A, B), 4))
        # Cosine: 先按样本求平均向量再算余弦, 作为参考基线
        a_mean = A.mean(dim=0)
        b_mean = B.mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(
            a_mean.unsqueeze(0), b_mean.unsqueeze(0)
        ).item()
        cos_list.append(round(cos, 4))

    return {'cka': cka_list, 'cosine': cos_list}


# ══════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════

def plot_representation_similarity(all_sim_results):
    """Fig 14: 逐层表征相似度 — 主线 Linear CKA, 虚线参考 mean-cosine."""
    fig, ax = plt.subplots(figsize=(10, 5))

    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    cmap = plt.cm.tab10

    all_cka_vals = []
    for i, (pair_name, sims) in enumerate(all_sim_results.items()):
        # 兼容旧格式 (list) 与新格式 (dict)
        if isinstance(sims, dict):
            cka = sims.get('cka', [])
            cos = sims.get('cosine', [])
        else:
            cka, cos = sims, []
        layers = list(range(len(cka)))
        ax.plot(layers, cka, marker=markers[i % len(markers)], linestyle='-',
                label=f'{pair_name} (CKA)', linewidth=2, markersize=7,
                color=cmap(i))
        if cos:
            ax.plot(list(range(len(cos))), cos, marker=markers[i % len(markers)],
                    linestyle=':', linewidth=1.2, markersize=4,
                    alpha=0.55, color=cmap(i),
                    label=f'{pair_name} (cos, ref)')
        all_cka_vals.extend(cka)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Similarity (Linear CKA, solid)')
    ax.set_title('Fig 14. Layer-wise Representation Similarity (CKA)',
                 fontsize=14, fontweight='bold')
    y_min = min(all_cka_vals) - 0.02 if all_cka_vals else 0
    ax.set_ylim(max(y_min, 0), 1.01)
    ax.grid(True, alpha=0.3)

    # Laitinen et al. 文献参考 (同为 CKA, 数值可直接对照)
    if all_cka_vals:
        n_layers = max(
            len(s['cka']) if isinstance(s, dict) else len(s)
            for s in all_sim_results.values()
        )
        mid_x = n_layers // 2
        ax.annotate(
            'Ref: Laitinen et al.\nCKA drop 0.32–0.47\n(intermediate layers)',
            xy=(mid_x, y_min + 0.01), fontsize=7.5, color='gray', alpha=0.75,
            ha='center', va='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', ec='gray', alpha=0.5))

    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, 'fig14_representation_similarity.png')


def plot_layer_drift_heatmap(all_drift_results):
    """Fig 13: 层级漂移热力图 [v6 ENHANCED: 双重分组 — 功能(Attn/FFN) × 深度(Shallow/Deep)]"""
    pair_names = list(all_drift_results.keys())
    n_layers = max(
        len(s['per_layer']) for s in all_drift_results.values()
    )

    # --- 主热力图: per-layer drift ---
    data = np.zeros((len(pair_names), n_layers))
    for i, (name, summary) in enumerate(all_drift_results.items()):
        for layer, drift in summary['per_layer'].items():
            data[i, layer] = drift

    fig, axes = plt.subplots(2, 1, figsize=(12, 4 + len(pair_names) * 2),
                             gridspec_kw={'height_ratios': [3, 2]})

    # Top: per-layer heatmap
    ax = axes[0]
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels(pair_names)
    ax.set_xlabel('Layer')
    ax.set_title('Fig 13. Parameter Drift Heatmap (Per Layer)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Relative Drift', shrink=0.8)

    for i in range(len(pair_names)):
        for j in range(n_layers):
            if data[i, j] > 0:
                ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center', fontsize=7)

    # [v6 NEW] Bottom: Attn vs FFN × Shallow vs Deep grouped bar
    ax2 = axes[1]
    func_groups = ['attn', 'ffn']
    depth_groups = ['Shallow', 'Deep']
    x = np.arange(len(pair_names))
    bar_width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors_map = {
        ('attn', 'Shallow'): '#4e79a7', ('attn', 'Deep'): '#a0cbe8',
        ('ffn', 'Shallow'): '#f28e2b', ('ffn', 'Deep'): '#ffbe7d',
    }

    for idx, (func, depth) in enumerate([(f, d) for f in func_groups for d in depth_groups]):
        vals = []
        for pair_name, summary in all_drift_results.items():
            half = n_layers // 2
            # Use actual per-category per-layer data
            if func == 'attn':
                layer_data = summary.get('attn_per_layer', {})
            else:
                layer_data = summary.get('ffn_per_layer', {})

            if layer_data:
                if depth == 'Shallow':
                    sel = [v for k, v in layer_data.items() if k < half]
                else:
                    sel = [v for k, v in layer_data.items() if k >= half]
                vals.append(np.mean(sel) if sel else 0)
            else:
                # Fallback: use per_category overall (no depth split available)
                per_cat = summary.get('per_category', {})
                vals.append(per_cat.get(func, 0))

        label = f'{func.upper()} ({depth})'
        ax2.bar(x + offsets[idx] * bar_width, vals, bar_width,
                label=label, color=colors_map.get((func, depth), '#999'))

    ax2.set_xticks(x)
    ax2.set_xticklabels(pair_names, fontsize=9)
    ax2.set_ylabel('Avg Drift')
    ax2.set_title('Attn vs FFN × Shallow vs Deep', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, ncol=4, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Fig 13. Parameter Drift Analysis', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_figure(fig, 'fig13_parameter_drift_heatmap.png')


# ══════════════════════════════════════════════════════════
# 4.3 Cross-Modal Alignment Metrics (VLM)
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_cross_modal_alignment(model, tokenizer, test_images):
    """配对余弦相似度 + 跨模态检索准确率"""
    from PIL import Image

    text_prompts = [img.get('description') or '一张图片' for img in test_images[:10]]
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
        text = img_info.get('description') or '一张图片'
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
        'effective': bool(improvement > 0.05),
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
    image_token_len = getattr(model.config, 'image_token_len', 64)
    image_special_token = getattr(model.config, 'image_special_token', '<|image_pad|>')
    image_placeholder = image_special_token * image_token_len

    for img_info in test_images[:3]:
        prompt = f'{image_placeholder}请描述这张图片。'
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
# [v6 NEW] 4.6.1 因果推断逻辑
# ══════════════════════════════════════════════════════════

def causal_inference_backbone(backbone_result, forgetting_data=None):
    """[v6 ENHANCED] 结合 M2 遗忘结果与 M4 backbone drift 做因果推断。

    判定逻辑 (来源: VLM-CL综述 + Laitinen + Yao et al.):
    - 遗忘严重 + 浅层漂移大 → 共享模块干扰 (VLM-CL 失败模式 2)
    - 遗忘不严重 + 漂移集中在深层 → 正常多模态适配
    - 遗忘严重 + 漂移均匀 → 全局性干扰
    """
    if not backbone_result:
        return None

    pattern = backbone_result.get('drift_pattern', 'uniform')
    shallow = backbone_result.get('shallow_avg_drift', 0)
    deep = backbone_result.get('deep_avg_drift', 0)

    # 加载 M2 遗忘数据
    if forgetting_data is None:
        from diagnostic_utils import load_json
        m2 = load_json('retention_matrix.json') or {}
        cross_modal = m2.get('cross_modal_forgetting', {})
        quality_drop = cross_modal.get('quality_drop', 0)
    else:
        quality_drop = forgetting_data.get('quality_drop', 0)

    forgetting_severe = quality_drop >= 0.5

    if forgetting_severe and pattern == 'shallow_dominant':
        diagnosis = '共享模块干扰（VLM-CL 失败模式 2）'
        mechanism = '低层 attention heads 扰动（Laitinen & Imanov, 2026）'
        recommendation = '冻结浅层 / 使用 O-LoRA（VLM-CL 综述）'
        severity = 'WARNING'
    elif forgetting_severe and pattern == 'deep_dominant':
        diagnosis = '深层表征重组超出正常范围'
        mechanism = '概念电路重组（Yao et al., 2025）但过度干扰'
        recommendation = '降低学习率 / 增加文本混合数据'
        severity = 'WARNING'
    elif not forgetting_severe and pattern == 'deep_dominant':
        diagnosis = '正常的多模态适配'
        mechanism = '深层适应新模态（预期行为）'
        recommendation = '无需干预'
        severity = 'PASS'
    elif forgetting_severe and pattern == 'uniform':
        diagnosis = '全局性干扰'
        mechanism = '学习率过高或数据分布偏移过大'
        recommendation = '降低学习率 / 增加文本混合数据 / OGPSA梯度正交投影'
        severity = 'WARNING'
    else:
        diagnosis = '轻微参数漂移，在可接受范围内'
        mechanism = '正常训练过程中的参数更新'
        recommendation = '无需干预，可继续训练'
        severity = 'PASS'

    return {
        'diagnosis': diagnosis,
        'mechanism': mechanism,
        'recommendation': recommendation,
        'severity': severity,
        'forgetting_severe': forgetting_severe,
        'quality_drop': round(quality_drop, 2),
        'drift_pattern': pattern,
        'shallow_drift': round(shallow, 4),
        'deep_drift': round(deep, 4),
    }


# ══════════════════════════════════════════════════════════
# 可视化: Fig 15 – Cross-Modal Alignment (Dimensionality Reduction)
# ══════════════════════════════════════════════════════════

def plot_cross_modal_tsne(alignment_before, alignment_after):
    """Fig 15: 散点图 — image vs text embeddings in 2D (t-SNE preferred, PCA fallback for small N)"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    def _reduce_2d(all_embeds):
        """Use t-SNE when enough samples, PCA fallback for small N."""
        n_samples = all_embeds.shape[0]
        if n_samples >= 6:
            perplexity = min(30, max(2, n_samples // 2 - 1))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                        n_iter=1000, init='pca', learning_rate='auto')
            return tsne.fit_transform(all_embeds), 't-SNE'
        else:
            pca = PCA(n_components=2)
            return pca.fit_transform(all_embeds), 'PCA'

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

        all_embeds = torch.cat([img_embeds, txt_embeds], dim=0).float().numpy()
        n_img = img_embeds.shape[0]
        n_samples = all_embeds.shape[0]

        coords, method = _reduce_2d(all_embeds)

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
        return method

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    method_a = _plot_panel(axes[0], alignment_before, 'VLM-Pretrain')
    method_b = _plot_panel(axes[1], alignment_after, 'VLM-SFT')
    method = method_a or method_b or 't-SNE'
    fig.suptitle(f'Fig 15. Cross-Modal Alignment ({method})', fontsize=14, fontweight='bold')
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
    # Per-layer value annotations
    for i, val in enumerate(interactions):
        ax.annotate(f'{val:.3f}', (i, val), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7.5, color='#333')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Image-Text Interaction Strength')
    ax.set_title('Fig 16. Visual Information Flow', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Dynamic y-axis based on data range
    y_max = max(interactions) if interactions else 1.0
    y_min = min(interactions) if interactions else 0
    margin = max((y_max - y_min) * 0.25, 0.02)
    ax.set_ylim(max(y_min - margin, 0), y_max + margin)

    trend = flow_result.get('trend', '')
    # Annotate min/max layers
    max_layer = int(np.argmax(interactions))
    min_layer = int(np.argmin(interactions))
    ax.annotate(f'Peak: L{max_layer}', (max_layer, interactions[max_layer]),
                textcoords='offset points', xytext=(15, -5), fontsize=9,
                fontweight='bold', color='#e15759',
                arrowprops=dict(arrowstyle='->', color='#e15759', lw=1.2))
    ax.annotate(f'Min: L{min_layer}', (min_layer, interactions[min_layer]),
                textcoords='offset points', xytext=(15, 5), fontsize=9,
                fontweight='bold', color='#4e79a7',
                arrowprops=dict(arrowstyle='->', color='#4e79a7', lw=1.2))

    ax.annotate(f'Trend: {trend}\nRange: {y_min:.4f} – {y_max:.4f}',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold', va='top',
                color='green' if trend == 'increasing' else 'orange',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    save_figure(fig, 'fig16_visual_info_flow.png')


# ══════════════════════════════════════════════════════════
# 可视化: Fig 17 – Backbone Drift: Shallow vs Deep
# ══════════════════════════════════════════════════════════

def plot_backbone_drift(backbone_result, causal_result=None):
    """Fig 17: 柱状图 — shallow vs deep layers drift from VLM training
    [v6 ENHANCED] 标注 VLM-CL 失败模式归因"""
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

    # [v6 NEW] 因果推断标注
    if causal_result:
        diag = causal_result.get('diagnosis', '')
        sev = causal_result.get('severity', '')
        color = '#e15759' if sev == 'WARNING' else '#59a14f'
        ax.annotate(f'诊断: {diag}\n建议: {causal_result.get("recommendation", "")}',
                    xy=(0.5, 0.02), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec=color, alpha=0.9))

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
    plot_layer_drift_heatmap(all_drift)

    # ---- 4.2 Representation Similarity ----
    print("\n[4.2] Representation Similarity...")
    all_sim = {}
    for stage_a, stage_b in STAGE_PAIRS:
        pair_name = f'{stage_a}→{stage_b}'
        print(f"  Computing {pair_name}...")
        sims = layer_representation_similarity(stage_a, stage_b, tokenizer)
        all_sim[pair_name] = sims
        print(f"    CKA   : {[f'{s:.3f}' for s in sims['cka']]}")
        print(f"    cosine: {[f'{s:.3f}' for s in sims['cosine']]}")

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

            # [v6 NEW] 因果推断
            print("\n[4.6.1] Causal Inference (combining M2 forgetting + M4 drift)...")
            causal = causal_inference_backbone(backbone)
            if causal:
                results['causal_inference'] = causal
                print(f"    Diagnosis: {causal['diagnosis']}")
                print(f"    Severity: {causal['severity']}")
                print(f"    Recommendation: {causal['recommendation']}")
            else:
                causal = None

            plot_backbone_drift(backbone, causal)
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

    print("\n  Representation Similarity (min CKA per transition):")
    for pair_name, sims in all_sim.items():
        cka_list = sims['cka'] if isinstance(sims, dict) else sims
        min_sim = min(cka_list)
        min_layer = cka_list.index(min_sim)
        print(f"    {pair_name}: min_CKA={min_sim:.4f} @ layer {min_layer}")

    return results


def _load_test_images():
    """加载VLM测试图片（使用集中式加载器，含 ground truth 标注）"""
    return load_test_images(max_images=15)


if __name__ == '__main__':
    run_module4()
