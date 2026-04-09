"""
Module 1: Output Behavior Diagnosis — 批量推理脚本
对4个训练阶段的checkpoint，用同一组prompts做推理，收集输出用于评分。
"""
import os
import sys
import json
import torch
from pathlib import Path

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
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'test_prompts.json')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9


def load_model(ckpt_name):
    """加载模型和tokenizer"""
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{ckpt_name}.pth')
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.half().eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, stage_name):
    """生成单条回复"""
    if stage_name == 'pretrain':
        # Pretrain模型没有经过chat template训练，直接用raw text
        input_text = tokenizer.bos_token + prompt
    else:
        # SFT/GRPO/DPO使用chat template
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            inputs=inputs.input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的token
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)

    # 扁平化所有prompts
    all_prompts = []
    for category, items in prompt_data.items():
        for item in items:
            all_prompts.append({**item, 'category': category})

    print(f"共 {len(all_prompts)} 条prompt, {len(STAGES)} 个阶段")

    for stage_name, ckpt_name in STAGES.items():
        print(f"\n{'='*60}")
        print(f"Stage: {stage_name} ({ckpt_name})")
        print(f"{'='*60}")

        model, tokenizer = load_model(ckpt_name)
        results = []

        for p in all_prompts:
            print(f"  [{p['category']}] {p['prompt'][:30]}...", end=' ')
            response = generate_response(model, tokenizer, p['prompt'], stage_name)
            print(f"-> {len(response)} chars")

            results.append({
                'id': p['id'],
                'category': p['category'],
                'prompt': p['prompt'],
                'response': response,
                'stage': stage_name,
            })

        # 保存该阶段的结果
        output_path = os.path.join(RESULTS_DIR, f'{stage_name}_responses.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {output_path}")

        # 释放GPU显存
        del model
        torch.cuda.empty_cache()

    # 汇总所有结果到一个文件
    all_results = {}
    for stage_name in STAGES.keys():
        path = os.path.join(RESULTS_DIR, f'{stage_name}_responses.json')
        with open(path, 'r', encoding='utf-8') as f:
            all_results[stage_name] = json.load(f)

    summary_path = os.path.join(RESULTS_DIR, 'all_responses.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存到 {summary_path}")


if __name__ == '__main__':
    main()
