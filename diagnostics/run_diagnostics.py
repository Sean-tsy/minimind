"""
Training Diagnostic Runner — 主入口
串联四模块，生成完整诊断报告。

Usage:
    cd diagnostics && python run_diagnostics.py              # 运行全部
    cd diagnostics && python run_diagnostics.py --module 1   # 只运行 Module 1
    cd diagnostics && python run_diagnostics.py --module 1 3 # 运行 Module 1 和 3
"""
import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))
from diagnostic_utils import print_header, save_json, load_json, RESULTS_DIR, FIGURES_DIR


def run_all(modules=None):
    """运行诊断 pipeline"""
    start_time = time.time()
    all_results = {}
    modules = modules or [1, 2, 3, 4]

    if 1 in modules:
        from module1_stage_goal import run_module1
        all_results['module1'] = run_module1()

    if 2 in modules:
        from module2_retention import run_module2
        all_results['module2'] = run_module2()

    if 3 in modules:
        from module3_pathology import run_module3
        all_results['module3'] = run_module3()

    if 4 in modules:
        from module4_localization import run_module4
        all_results['module4'] = run_module4()

    elapsed = time.time() - start_time

    # Detect judge backend
    gemini_available = bool(os.environ.get('GEMINI_API_KEY'))
    judge_backend = 'gemini' if gemini_available else 'offline_rules'
    confidence = 'high' if gemini_available else 'medium'

    all_results['_meta'] = {
        'elapsed_seconds': round(elapsed, 1),
        'modules_run': modules,
        'judge_backend': judge_backend,
        'confidence': confidence,
    }

    save_json(all_results, 'diagnostic_report.json')
    print_final_report(all_results)
    generate_markdown_report(all_results)
    return all_results


def print_final_report(results):
    """打印最终诊断报告摘要"""
    print_header("FINAL DIAGNOSTIC REPORT")

    elapsed = results.get('_meta', {}).get('elapsed_seconds', 0)
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    judge = results.get('_meta', {}).get('judge_backend', 'unknown')
    conf = results.get('_meta', {}).get('confidence', 'unknown')
    print(f"  Judge backend: {judge}  |  Confidence: {conf}\n")

    # ---- Module 1 Summary ----
    m1 = results.get('module1', {})
    if m1:
        print("  Module 1: Stage Goal Verification")
        print("  " + "─" * 50)
        for key, val in m1.items():
            if isinstance(val, dict) and 'status' in val:
                metric = val.get('metric', key)
                stage = val.get('stage', '')
                score_key = next((k for k in ['avg_score', 'rate', 'harmful_refusal_rate'] if k in val), None)
                score = val.get(score_key, '') if score_key else ''
                print(f"    {stage:>8s} | {metric:<30s} | {str(score):<8s} | {val['status']}")
        print()

    # ---- Module 2 Summary ----
    m2 = results.get('module2', {})
    if m2:
        print("  Module 2: Capability Retention")
        print("  " + "─" * 50)
        matrix = m2.get('retention_matrix', {})
        for cap, scores in matrix.items():
            score_str = ' '.join(f'{s}:{v}' for s, v in scores.items())
            print(f"    {cap:<25s} | {score_str}")
        forgetting = m2.get('forgetting_rates', {})
        severe = []
        for cap, rates in forgetting.items():
            for trans, rate in rates.items():
                if rate > 0.15:
                    severe.append(f"{cap} {trans}: {rate:+.1%}")
        if severe:
            print(f"    ⚠️  Significant forgetting: {'; '.join(severe)}")
        print()

    # ---- Module 3 Summary ----
    m3 = results.get('module3', {})
    if m3:
        print("  Module 3: Pathology Detection")
        print("  " + "─" * 50)
        for key, val in m3.items():
            if isinstance(val, dict) and 'status' in val:
                print(f"    {val.get('pathology',''):<22s} @ {val.get('stage',''):<8s} | {val['status']}")
        print()

    # ---- Module 4 Summary ----
    m4 = results.get('module4', {})
    if m4:
        print("  Module 4: Change Localization")
        print("  " + "─" * 50)
        drift = m4.get('parameter_drift', {})
        for pair, summary in drift.items():
            print(f"    {pair:<20s} | pattern: {summary.get('drift_pattern','')}")
        sim = m4.get('representation_similarity', {})
        for pair, sims in sim.items():
            if isinstance(sims, list) and sims:
                min_s = min(sims)
                print(f"    {pair:<20s} | min layer sim: {min_s:.4f}")
        print()

    # ---- Key Findings ----
    findings = extract_key_findings(results)
    if findings:
        print("  KEY FINDINGS:")
        print("  " + "─" * 50)
        for i, f in enumerate(findings, 1):
            print(f"    {i}. {f}")
        print()

    print("═" * 60)
    print(f"  Full results saved to: {RESULTS_DIR}/diagnostic_report.json")
    print(f"  Figures saved to: {FIGURES_DIR}/")
    print("═" * 60)


def extract_key_findings(results):
    """从诊断结果中提取关键发现"""
    findings = []

    # Module 1
    m1 = results.get('module1', {})
    if 'pretrain_fluency' in m1:
        s = m1['pretrain_fluency']['avg_score']
        findings.append(f"Pretrain fluency score: {s}/5")

    pre_if = m1.get('pretrain_instruction_following', {}).get('rate', 0)
    sft_if = m1.get('sft_instruction_following', {}).get('rate', 0)
    if sft_if > 0 and pre_if >= 0:
        findings.append(
            f"SFT improved instruction following from {pre_if:.0%} to {sft_if:.0%}"
        )

    for stage in ['sft', 'grpo', 'dpo']:
        key = f'{stage}_alignment'
        if key in m1:
            hr = m1[key].get('harmful_refusal_rate', 0)
            findings.append(f"{stage.upper()} harmful refusal rate: {hr:.0%}")

    # [v6] Alignment tax (SFT baseline vs GRPO / DPO)
    atax = m1.get('alignment_tax', {})
    if atax:
        if 'avg_tax_grpo' in atax:
            findings.append(
                f"Alignment tax (SFT→GRPO): {atax.get('avg_tax_grpo', 0):+.2f} [{atax.get('status_grpo', '')}]"
            )
        if 'avg_tax_dpo' in atax:
            findings.append(
                f"Alignment tax (SFT→DPO): {atax.get('avg_tax_dpo', 0):+.2f} [{atax.get('status_dpo', '')}]"
            )

    # Module 2 - forgetting
    m2 = results.get('module2', {})
    forgetting = m2.get('forgetting_rates', {})
    for cap, rates in forgetting.items():
        for trans, rate in rates.items():
            if rate > 0.15:
                findings.append(f"⚠️ {cap} forgetting {trans}: {rate:+.1%}")

    # [v6 NEW] VLM-CL failure modes
    failure_modes = m2.get('vlm_cl_failure_modes', [])
    for fm in failure_modes:
        if fm.get('severity') not in ('PASS', None):
            findings.append(f"⚠️ VLM-CL: {fm['mode']} — {fm.get('signal', '')}")

    # Module 3 - pathologies
    m3 = results.get('module3', {})
    for key, val in m3.items():
        if isinstance(val, dict) and val.get('status') in ('WARN', 'FAIL'):
            findings.append(f"⚠️ Pathology detected: {val.get('pathology','')} @ {val.get('stage','')}")

    # [v6 NEW] Hallucination attribution
    hall_attr = m3.get('hallucination_attribution', {})
    if hall_attr:
        findings.append(f"Hallucination source: {hall_attr.get('primary_source', 'unknown')}")

    # Module 4 - drift patterns
    m4 = results.get('module4', {})
    drift = m4.get('parameter_drift', {})
    for pair, summary in drift.items():
        if summary.get('drift_pattern') == 'shallow_dominant':
            findings.append(f"⚠️ {pair}: shallow-dominant drift (may affect basic language)")

    # [v6 NEW] Causal inference
    causal = m4.get('causal_inference', {})
    if causal and causal.get('severity') == 'WARNING':
        findings.append(f"⚠️ Causal: {causal['diagnosis']}")

    return findings[:10]  # 最多显示10条


def generate_markdown_report(results):
    """生成 Markdown 格式的诊断报告到 report/diagnostic_report.md"""
    lines = []
    lines.append('# MiniMind Training Diagnostic Report\n')
    lines.append(f'Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    lines.append(f'*Framework Version: v6 (Literature-Enhanced)*\n')

    elapsed = results.get('_meta', {}).get('elapsed_seconds', 0)
    modules_run = results.get('_meta', {}).get('modules_run', [])
    judge_backend = results.get('_meta', {}).get('judge_backend', 'unknown')
    confidence = results.get('_meta', {}).get('confidence', 'unknown')
    lines.append(f'Modules run: {modules_run}  |  Total time: {elapsed:.0f}s\n')
    lines.append(f'Judge backend: **{judge_backend}**  |  Confidence: **{confidence}**\n')
    lines.append('---\n')

    # ---- Module 1 ----
    m1 = results.get('module1', {})
    if m1:
        lines.append('## Module 1: Stage Goal Verification\n')
        lines.append('| Stage | Metric | Score | Status |')
        lines.append('|-------|--------|-------|--------|')
        for key, val in m1.items():
            if isinstance(val, dict) and 'status' in val:
                metric = val.get('metric', key)
                stage = val.get('stage', '')
                score_key = next((k for k in ['avg_score', 'rate', 'harmful_refusal_rate'] if k in val), None)
                score = val.get(score_key, '') if score_key else ''
                lines.append(f'| {stage} | {metric} | {score} | {val["status"]} |')
        # [v6] Alignment tax (SFT baseline)
        atax = m1.get('alignment_tax', {})
        if atax:
            lines.append('')
            lines.append('### Alignment Tax (baseline = SFT)\n')
            lines.append(
                f'Average tax  GRPO: **{atax.get("avg_tax_grpo", 0):+.2f}** [{atax.get("status_grpo", "")}]  |  '
                f'DPO: **{atax.get("avg_tax_dpo", 0):+.2f}** [{atax.get("status_dpo", "")}]\n'
            )
            per_dim = atax.get('per_dimension', {})
            if per_dim:
                lines.append('| Dimension | SFT | GRPO | Tax(GRPO) | DPO | Tax(DPO) |')
                lines.append('|-----------|-----|------|-----------|-----|----------|')
                for dim, info in per_dim.items():
                    lines.append(
                        f'| {dim} | {info["sft"]} | {info["grpo"]} | {info["tax_grpo"]:+.2f} | '
                        f'{info["dpo"]} | {info["tax_dpo"]:+.2f} |'
                    )
        lines.append('')

    # ---- Module 2 ----
    m2 = results.get('module2', {})
    if m2:
        lines.append('## Module 2: Capability Retention\n')
        matrix = m2.get('retention_matrix', {})
        if matrix:
            first_cap = next(iter(matrix.values()))
            stages = list(first_cap.keys())
            lines.append('| Capability | ' + ' | '.join(s.upper() for s in stages) + ' |')
            lines.append('|' + '---|' * (len(stages) + 1))
            for cap, scores in matrix.items():
                row = ' | '.join(str(scores.get(s, 'n/a')) for s in stages)
                lines.append(f'| {cap} | {row} |')
        lines.append('')
        forgetting = m2.get('forgetting_rates', {})
        severe = [f'{cap} {t}: {r:+.1%}' for cap, rates in forgetting.items()
                  for t, r in rates.items() if r > 0.15]
        if severe:
            lines.append('**Significant Forgetting:**\n')
            for s in severe:
                lines.append(f'- ⚠️ {s}')
            lines.append('')
        # v6: Aggregate normalized forgetting
        agg = m2.get('aggregate_forgetting', {})
        if agg:
            lines.append('**Aggregate Normalized Forgetting (Luo et al.):**\n')
            lines.append('| Transition | NF (%) | |E| |')
            lines.append('|------------|--------|-----|')
            for trans, info in agg.items():
                pct = info['normalized_forgetting_pct']
                n = info['n_dimensions']
                flag = '⚠️' if pct > 15 else ''
                lines.append(f'| {trans} | {pct:+.1f}% | {n} | {flag}')
            lines.append('')
        # [v6 NEW] VLM-CL failure modes
        fm_list = m2.get('vlm_cl_failure_modes', [])
        if fm_list:
            lines.append('### VLM-CL Failure Mode Mapping\n')
            lines.append('| Mode | Severity | Signal | Recommendation |')
            lines.append('|------|----------|--------|----------------|')
            for fm in fm_list:
                lines.append(f'| {fm.get("mode","")} | {fm.get("severity","")} | '
                             f'{fm.get("signal","")} | {fm.get("recommendation","")} |')
            lines.append('')

    # ---- Module 3 ----
    m3 = results.get('module3', {})
    if m3:
        lines.append('## Module 3: Pathology Detection\n')
        lines.append('| Category | Pathology | Stage | Status |')
        lines.append('|----------|-----------|-------|--------|')
        for key, val in m3.items():
            if isinstance(val, dict) and 'pathology' in val:
                cat = 'VLM' if val['pathology'] in ('modality_shortcut', 'description_collapse',
                                                     'visual_hallucination', 'grounding_failure') else 'LLM'
                lines.append(f'| {cat} | {val["pathology"]} | {val.get("stage", "")} | {val["status"]} |')
        # [v6 NEW] Hallucination source attribution
        hall_attr = m3.get('hallucination_attribution', {})
        if hall_attr:
            lines.append('')
            lines.append(f'**Hallucination Source**: {hall_attr.get("primary_source", "unknown")}')
            lines.append(f'  - Evidence: {hall_attr.get("evidence", "")}')
            lines.append(f'  - Recommendation: {hall_attr.get("recommendation", "")}')
        # [v6 NEW] Spatial specificity
        grounding = m3.get('grounding_failure', {})
        if grounding and 'avg_spatial_specificity' in grounding:
            lines.append(f'  - Spatial Specificity: {grounding["avg_spatial_specificity"]:.3f}')
        lines.append('')

    # ---- Module 4 ----
    m4 = results.get('module4', {})
    if m4:
        lines.append('## Module 4: Change Localization\n')
        drift = m4.get('parameter_drift', {})
        if drift:
            lines.append('| Transition | Pattern | Shallow Avg | Deep Avg |')
            lines.append('|------------|---------|-------------|----------|')
            for pair, summary in drift.items():
                lines.append(f'| {pair} | {summary.get("drift_pattern", "")} '
                             f'| {summary.get("shallow_avg", 0):.4f} '
                             f'| {summary.get("deep_avg", 0):.4f} |')
        # [v6 NEW] Causal inference
        causal = m4.get('causal_inference', {})
        if causal:
            lines.append('')
            lines.append('### Causal Inference (M2 + M4)\n')
            lines.append(f'- **Diagnosis**: {causal.get("diagnosis", "")}')
            lines.append(f'- **Mechanism**: {causal.get("mechanism", "")}')
            lines.append(f'- **Recommendation**: {causal.get("recommendation", "")}')
            lines.append(f'- **Severity**: {causal.get("severity", "")}')
        lines.append('')

    # ---- Key Findings ----
    findings = extract_key_findings(results)
    if findings:
        lines.append('## Key Findings\n')
        for i, f in enumerate(findings, 1):
            lines.append(f'{i}. {f}')
        lines.append('')

    # ---- Write ----
    report_dir = os.path.join(os.path.dirname(__file__), '..', 'report')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'diagnostic_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Markdown report saved to: {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MiniMind Training Diagnostic')
    parser.add_argument('--module', nargs='+', type=int, default=None,
                        help='Module numbers to run (1-4). Default: all')
    args = parser.parse_args()
    run_all(modules=args.module)
