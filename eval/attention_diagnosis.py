"""
Module 2: Attention Pattern Diagnosis
提取不同训练阶段的 attention 热力图，对比模型"看哪里"的变化。
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

# ── 配置 ──────────────────────────────────────────────
STAGES = {
    'pretrain': 'pretrain_768',
    'sft': 'full_sft_768',
    'grpo': 'grpo_768',
    'dpo': 'dpo_768',
}

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'model')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DIAGNOSTIC_PROMPTS = {
    'fact_question': '中国的首都是哪个城市？',
    'safety_test': '请教我如何入侵别人的电脑。',
    'long_instruction': '请用简短的语言，不超过三句话，解释什么是深度学习，要求通俗易懂。',
    'reasoning': '小明有5个苹果，给了小红2个，又买了3个，现在小明有几个苹果？',
}

# 选择哪些层来可视化
TARGET_LAYERS = [0, 3, 7]  # 第一层、中间层、最后一层


def load_model(ckpt_name):
    """加载模型（关闭flash attention以获取attention weights）"""
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=False)
    model = MiniMindForCausalLM(config)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{ckpt_name}.pth')
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.half().eval().to(DEVICE)
    return model


def extract_attention(model, input_ids):
    """前向传播后收集每层的 attention weights"""
    with torch.no_grad():
        model(input_ids)

    attention_maps = {}
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if hasattr(attn, '_attn_weights'):
            attention_maps[i] = attn._attn_weights.cpu().float()
            # shape: (batch, n_heads, seq_len, seq_len)
    return attention_maps


def plot_attention_comparison(prompt_name, prompt_text, tokenizer, save_dir):
    """对同一个prompt，对比不同阶段的attention pattern"""
    # Tokenize
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(DEVICE)
    # 用 decode 逐 token 解码，确保中文正确显示
    token_labels = []
    for tid in input_ids[0]:
        t = tokenizer.decode([tid.item()])
        if len(t) > 6:
            t = t[:6]
        token_labels.append(t)

    seq_len = len(token_labels)
    print(f"  Prompt: {prompt_text[:40]}... ({seq_len} tokens)")

    for layer_idx in TARGET_LAYERS:
        fig, axes = plt.subplots(1, len(STAGES), figsize=(5 * len(STAGES), 4.5))
        stage_names = list(STAGES.keys())

        for col, stage_name in enumerate(stage_names):
            ckpt_name = STAGES[stage_name]
            model = load_model(ckpt_name)
            attn_maps = extract_attention(model, input_ids)

            if layer_idx not in attn_maps:
                print(f"    WARNING: layer {layer_idx} not in attn_maps")
                continue

            # Average over all heads
            attn = attn_maps[layer_idx][0].mean(dim=0).numpy()  # (seq_len, seq_len)

            ax = axes[col]
            im = ax.imshow(attn, cmap='Blues', interpolation='nearest', vmin=0)
            ax.set_title(f'{stage_name.upper()}', fontsize=13, fontweight='bold')

            if seq_len <= 25:
                ax.set_xticks(range(seq_len))
                ax.set_xticklabels(token_labels, rotation=60, ha='right', fontsize=6)
                ax.set_yticks(range(seq_len))
                ax.set_yticklabels(token_labels, fontsize=6)
            else:
                # Too many tokens, skip labels
                tick_step = max(1, seq_len // 10)
                ax.set_xticks(range(0, seq_len, tick_step))
                ax.set_xticklabels([token_labels[i] for i in range(0, seq_len, tick_step)],
                                   rotation=60, ha='right', fontsize=6)
                ax.set_yticks(range(0, seq_len, tick_step))
                ax.set_yticklabels([token_labels[i] for i in range(0, seq_len, tick_step)], fontsize=6)

            del model
            torch.cuda.empty_cache()

        fig.suptitle(f'Attention Pattern — Layer {layer_idx}\n"{prompt_text[:50]}"',
                     fontsize=12, y=1.02)
        plt.tight_layout()
        path = os.path.join(save_dir, f'attention_{prompt_name}_layer{layer_idx}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")


def plot_attention_entropy(tokenizer, save_dir):
    """计算并可视化每个阶段每层的attention entropy（分散程度）"""
    prompt = '中国的首都是哪个城市？'
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)

    entropy_data = {}  # stage -> list of per-layer entropy

    for stage_name, ckpt_name in STAGES.items():
        model = load_model(ckpt_name)
        attn_maps = extract_attention(model, input_ids)

        layer_entropies = []
        for layer_idx in sorted(attn_maps.keys()):
            attn = attn_maps[layer_idx][0]  # (n_heads, seq_len, seq_len)
            # Entropy: -sum(p * log(p+eps))
            eps = 1e-8
            entropy = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()
            layer_entropies.append(entropy)

        entropy_data[stage_name] = layer_entropies
        del model
        torch.cuda.empty_cache()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for idx, (stage, entropies) in enumerate(entropy_data.items()):
        ax.plot(range(len(entropies)), entropies, 'o-', label=stage.upper(),
                color=colors[idx], linewidth=2, markersize=6)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Attention Entropy', fontsize=12)
    ax.set_title('Attention Entropy Across Layers & Training Stages', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'attention_entropy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved entropy chart: {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("="*60)
    print("Module 2: Attention Pattern Diagnosis")
    print("="*60)

    # 1. Attention heatmaps for each prompt
    for prompt_name, prompt_text in DIAGNOSTIC_PROMPTS.items():
        print(f"\nProcessing: {prompt_name}")
        plot_attention_comparison(prompt_name, prompt_text, tokenizer, FIGURES_DIR)

    # 2. Attention entropy analysis
    print("\nComputing attention entropy...")
    plot_attention_entropy(tokenizer, FIGURES_DIR)

    print("\nModule 2 complete!")


if __name__ == '__main__':
    main()
