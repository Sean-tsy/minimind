<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">
  <h3>"Loss curves are 1D signals, but training problems are multi-dimensional"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

---

# MiniMind Training Diagnostic Framework

> A multi-stage training diagnostic framework built on [MiniMind](https://github.com/jingyaogong/minimind) (LLM) and [MiniMind-V](https://github.com/jingyaogong/minimind-v) (VLM).
> Expands the 1D loss curve into a multi-dimensional diagnostic dashboard with **17 diagnostic charts**, backed by 12 research papers.

## 📌 Motivation

In a multi-stage LLM/VLM training pipeline (Pretrain → SFT → GRPO → DPO → VLM-Pretrain → VLM-SFT), the only signal available to the practitioner is the loss curve. But a decreasing loss **cannot tell you**:

| Blind Spot | Example |
|------------|---------|
| 🎯 Did this stage achieve its goal? | Loss is decreasing, but the model learned wrong patterns |
| 💥 Did the new stage break previous gains? | SFT reduced loss, but pretrain knowledge was overwritten |
| 🦠 Any known pathological behaviors? | Over-alignment, modality shortcut, hallucination — none show up in loss |
| 🔧 Where should you fix? | Which layers to freeze? Which stage's data to adjust? |

This framework systematizes these blind spots into **4 diagnostic modules**, outputting structured PASS / WARNING / FAIL verdicts with actionable optimization suggestions.

### Relationship to Original Repos

| | [MiniMind](https://github.com/jingyaogong/minimind) / [MiniMind-V](https://github.com/jingyaogong/minimind-v) | This Project |
|---|---|---|
| **Purpose** | Training tutorial: how to run the pipeline | Diagnostic tool: how good is the trained model |
| **Input** | Datasets + training scripts | Trained checkpoints |
| **Output** | Weight files + loss curves | 17 diagnostic charts + structured report |
| **Role** | Engine (MLE) | Dashboard (DS) |
| **Coverage** | Single-stage training | Cross-stage diagnosis + cross-modal forgetting |
| **Academic Depth** | No citations | 12-paper methodology |

---

## 📌 Framework Overview

```
┌──────────────────────────────────────────────────────────────────┐
│              LLM/VLM Training Diagnostic Framework               │
│                                                                  │
│  Input: 6 stage checkpoints + text prompts + image-text pairs    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Module 1: Stage Goal Verification                               │
│  ├── LLM: fluency / instruction following / alignment precision  │
│  ├── Alignment tax: per-dimension capability tracking            │
│  └── VLM: visual description / VQA / visual instruction          │
│                                                                  │
│  Module 2: Capability Retention                                  │
│  ├── LLM→LLM: Does SFT forget pretrain knowledge?               │
│  ├── LLM→VLM: Does VLM training break text ability?             │
│  ├── VLM→VLM: Does VLM-SFT break basic visual description?      │
│  └── Normalized forgetting (Luo et al.) + VLM-CL failure modes   │
│                                                                  │
│  Module 3: Pathology Detection                                   │
│  ├── LLM: repetition / format overfitting / over-alignment /     │
│  │        mode collapse                                          │
│  ├── VLM: modality shortcut / description collapse /             │
│  │        visual hallucination / grounding failure                │
│  └── Hallucination source attribution (encoder/projection/LLM)   │
│                                                                  │
│  Module 4: Change Localization                                   │
│  ├── Parameter drift (Attn Q/K/V vs FFN × shallow vs deep)       │
│  ├── Representation similarity + CKA calibration                 │
│  └── VLM: cross-modal alignment / info flow / backbone drift     │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  Output: 17 charts + PASS/WARN/FAIL verdicts + recommendations   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📌 Models Under Diagnosis

### MiniMind LLM

MiniMind is an ultra-small language model trained entirely from scratch, architecturally aligned with the Qwen3 ecosystem. Configuration used: `hidden_size=768, num_hidden_layers=8`, ~64M parameters.

| Stage | Checkpoint | Training Goal |
|-------|-----------|---------------|
| Pretrain | `pretrain_768.pth` | Learn language patterns and world knowledge |
| SFT | `full_sft_768.pth` | Learn to follow instructions |
| GRPO | `grpo_768.pth` | RL-based preference alignment |
| DPO | `dpo_768.pth` | Direct preference optimization |

### MiniMind-V VLM

MiniMind-V extends MiniMind with visual capabilities via SigLIP2 vision encoder + MLP Projection (reshape compression 256→64 tokens), ~67M parameters.

| Stage | Checkpoint | Training Goal |
|-------|-----------|---------------|
| VLM Pretrain | `vlm_pretrain_768.pth` | Vision-language alignment (projection only) |
| VLM SFT | `vlm_sft_768.pth` | Visual instruction following (full fine-tuning) |

---

## 📌 Quick Start

### Prerequisites

```bash
git clone https://github.com/your-repo/minimind-diagnostic
cd minimind-diagnostic
pip install -r requirements.txt
```

### Prepare Checkpoints

Place trained weights in `checkpoints/`:

```
checkpoints/
├── pretrain_768.pth
├── full_sft_768.pth
├── grpo_768.pth
├── dpo_768.pth
├── vlm_pretrain_768.pth      # optional
└── vlm_sft_768.pth           # optional
```

> Without VLM checkpoints, the framework automatically skips VLM diagnostics.

### Run Diagnostics

```bash
cd diagnostics

# Run all 4 modules
python run_diagnostics.py

# Or specific modules
python run_diagnostics.py --module 1      # Module 1 only
python run_diagnostics.py --module 1 3    # Module 1 and 3
```

### With Gemini API (Recommended)

Setting a Gemini API key enables more accurate semantic evaluation (labeled `high confidence`).
Without API, the framework falls back to rule-based scoring (labeled `medium confidence`).

```bash
export GEMINI_API_KEY="your-api-key"
python run_diagnostics.py
```

### Output Files

```
results/
├── raw/                          # Structured JSON results
│   ├── diagnostic_report.json
│   ├── stage_comparison.json
│   ├── retention_matrix.json
│   ├── pathology_results.json
│   └── drift_analysis.json
└── figures/                      # 17 diagnostic charts (fig01~fig17.png)

report/
├── diagnostic_report.md          # Markdown diagnostic report
└── literature_review.md          # Literature-driven narrative
```

---

## 📌 17 Diagnostic Charts

### Module 1: Stage Goal Verification

| # | Chart | Question Answered |
|---|-------|-------------------|
| Fig 1 | Stage Goal Dashboard | Did each stage meet its goal? |
| Fig 2 | Alignment Precision Scatter | Is DPO precise or excessive? |
| Fig 3 | VLM Stage Comparison | How much did VLM-SFT improve over Pretrain? |
| Fig 4 | Behavior Transition | How did behavior patterns evolve? (5 classes) |

<div align="center">
<img src="./results/figures/fig01_goal_dashboard.png" width="90%"/>
<p><i>Fig 1. Stage Goal Dashboard</i></p>
</div>

<div align="center">
<img src="./results/figures/fig04_behavior_transition.png" width="70%"/>
<p><i>Fig 4. Behavior Transition — from Pretrain continuation to SFT instruction-following to RLHF safety refusal</i></p>
</div>

### Module 2: Capability Retention

| # | Chart | Question Answered |
|---|-------|-------------------|
| Fig 5 | Capability Retention Heatmap | Which capabilities were forgotten? |
| Fig 6 | Forgetting Rate Waterfall | How severe is forgetting? (normalized) |
| Fig 7 | Cross-Modal Forgetting | Did VLM break text ability? (with sub-scores) |

<div align="center">
<img src="./results/figures/fig05_retention_heatmap.png" width="80%"/>
<p><i>Fig 5. Capability Retention Heatmap — factual knowledge drops 50% after SFT</i></p>
</div>

### Module 3: Pathology Detection

| # | Chart | Question Answered |
|---|-------|-------------------|
| Fig 8 | Repetition Rate Trend | Is repetition improving or worsening? |
| Fig 9 | Paraphrase Consistency | Did SFT truly learn or just memorize? (decomposed) |
| Fig 10 | Mode Collapse Matrix | Did RLHF homogenize responses? |
| Fig 11 | Visual Dependency Scores | Is the VLM actually looking at images? |
| Fig 12 | Hallucination & Grounding | Where does the VLM hallucinate? |

<div align="center">
<img src="./results/figures/fig11_visual_dependency.png" width="70%"/>
<p><i>Fig 11. Visual Dependency — dependency score 0.781 (PASS), model is genuinely using visual input</i></p>
</div>

### Module 4: Change Localization

| # | Chart | Question Answered |
|---|-------|-------------------|
| Fig 13 | Parameter Drift Heatmap | Which layers/components changed most? |
| Fig 14 | Representation Similarity | Are changes in shallow or deep layers? |
| Fig 15 | Cross-Modal Alignment t-SNE | Did alignment actually succeed? |
| Fig 16 | Visual Information Flow | Where do visual features fuse? |
| Fig 17 | Backbone Drift Comparison | What did VLM training change in the LLM? |

<div align="center">
<img src="./results/figures/fig13_parameter_drift_heatmap.png" width="80%"/>
<p><i>Fig 13. Parameter Drift — Pretrain→SFT shows largest change (FFN: 0.13), RLHF stages minimal</i></p>
</div>

---

## 📌 Diagnostic Results Summary

Results from the actual MiniMind (768-dim, 8-layer) full pipeline diagnosis:

### Key Findings

1. **Factual knowledge catastrophic forgetting**: 50% forgetting after SFT (pretrain 2.0 → sft 1.0), far exceeding the 15% threshold suggested by Luo et al.

2. **DPO safety regression**: GRPO raised harmful refusal to 75% (PASS), but DPO degraded it to 25% (FAIL) — consistent with alignment tax theory (OGPSA).

3. **VLM shared-module interference**: Shallow drift (0.112) >> deep drift (0.008) during VLM training, matching Laitinen's lower-layer attention head vulnerability findings.

4. **High hallucination but genuine visual dependency**: Visual dependency 0.781 (PASS, model uses images), but hallucination rate 87.5% (FAIL). Source: visual encoder representation quality.

5. **Unexpected: VLM-SFT improves text ability**: VLM-SFT text quality (2.25) > pure SFT text quality (1.0) — multimodal training data may act as data augmentation at small scale.

---

## 📌 Literature Foundation

| Paper | Application |
|-------|-------------|
| MaP (Wang et al., 2025) | Evaluation instability theory |
| Luo et al. (2308.08747) | Normalized forgetting formula (M2) |
| Laitinen & Imanov (2601.18699) | Lower-layer attention head vulnerability (M4) |
| OGPSA (2602.07892) | Alignment tax quantification (M1) |
| Yao et al. | Concept circuit learning-forgetting trade-off (M4) |
| VLM-CL Survey (2508.04227) | Three VLM-CL failure modes (M2) |
| Jing et al. (2505.01958) | Hallucination source attribution (M3) |
| Grounding Survey (2509.10345) | Visual grounding analysis (M3) |

---

## 📌 Project Structure

```
minimind-diagnostic/
├── diagnostics/                    ← Core diagnostic code
│   ├── module1_stage_goal.py       # M1: Stage goal verification
│   ├── module2_retention.py        # M2: Capability retention
│   ├── module3_pathology.py        # M3: Pathology detection
│   ├── module4_localization.py     # M4: Change localization
│   ├── diagnostic_utils.py         # Utilities (inference/Gemini/IO)
│   ├── run_diagnostics.py          # Main entry point
│   └── test_prompts.json
├── model/                          ← Model definitions
│   ├── model_minimind.py           # MiniMind LLM
│   ├── model_vlm.py                # MiniMind-V VLM
│   └── siglip2-base-p16-ve/        # SigLIP2 vision encoder
├── trainer/                        ← Training scripts
├── checkpoints/                    ← Stage weights
├── results/
│   ├── raw/                        # JSON results
│   └── figures/                    # 17 diagnostic charts
├── report/
│   ├── diagnostic_report.md
│   └── literature_review.md
└── requirements.txt
```

---

## 📌 Requirements

| Item | Requirement |
|------|-------------|
| Python | ≥ 3.10 |
| GPU | Any CUDA GPU (~2GB VRAM, inference only) |
| Core | PyTorch, Transformers, scikit-learn, matplotlib |
| Optional | google-genai (Gemini API scoring) |
| Runtime | ~35 min (all 4 modules with VLM) |

---

## 📌 License

This project is open-sourced under the [Apache 2.0](./LICENSE) license.

---

## 📌 Acknowledgments

- [MiniMind](https://github.com/jingyaogong/minimind) — LLM training tutorial and model code
- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — VLM training tutorial and visual extension
- [SigLIP2](https://huggingface.co/jingyaogong/siglip2-base-p16-ve) — Vision encoder
- 12 research papers underpinning the diagnostic methodology (see [literature review](./report/literature_review.md))
