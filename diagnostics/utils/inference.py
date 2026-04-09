"""
Batch Inference Utility — 独立批量推理脚本
对4个训练阶段的checkpoint，用同一组prompts做推理，收集输出。

复用 diagnostic_utils 中的模型加载和生成函数，避免代码重复。

Usage:
    cd diagnostics && python -m utils.inference
"""
import os
import sys
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diagnostic_utils import (
    load_model, load_tokenizer, generate_response,
    STAGES, RESULTS_DIR, ROOT_DIR,
)

PROMPTS_FILE = os.path.join(ROOT_DIR, 'diagnostics', 'test_prompts.json')


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
    tokenizer = load_tokenizer()

    for stage_name, ckpt_name in STAGES.items():
        print(f"\n{'='*60}")
        print(f"Stage: {stage_name} ({ckpt_name})")
        print(f"{'='*60}")

        model = load_model(ckpt_name)
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
