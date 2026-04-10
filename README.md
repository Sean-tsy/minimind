<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">
  <h3>"Loss曲线是一维信号，但训练中的问题是多维的"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

---

# MiniMind Training Diagnostic Framework

> 基于 [MiniMind](https://github.com/jingyaogong/minimind)（LLM）与 [MiniMind-V](https://github.com/jingyaogong/minimind-v)（VLM）的多阶段训练诊断框架。
> 用 **17 张诊断图表** 将一维 loss 曲线扩展为多维诊断面板，12 篇论文支撑的诊断方法论。

## 📌 项目定位

多阶段 LLM/VLM 训练 pipeline（Pretrain → SFT → GRPO → DPO → VLM-Pretrain → VLM-SFT）中，训练者唯一能看到的是 loss 曲线。但 loss 下降**无法告诉你**：

| 盲区 | 举例 |
|------|------|
| 🎯 这个阶段达到目的了吗？ | loss 在降，但模型在学错误的 pattern |
| 💥 新阶段破坏了旧成果吗？ | SFT 让 loss 下降了，但 pretrain 知识被覆盖了 |
| 🦠 出现已知病态行为了吗？ | over-alignment、modality shortcut、hallucination——都不体现在 loss 上 |
| 🔧 出了问题该改哪里？ | 该冻结哪些层？该调哪个阶段的数据？ |

本框架将这些盲区系统化为 **4 个诊断模块**，输出结构化的 PASS / WARNING / FAIL 判定与可执行的优化建议。

### 与原 repo 的关系

| | [MiniMind](https://github.com/jingyaogong/minimind) / [MiniMind-V](https://github.com/jingyaogong/minimind-v) | 本项目 |
|---|---|---|
| **定位** | 训练教程：教你怎么跑 pipeline | 诊断工具：告诉你跑出来的模型好不好 |
| **输入** | 数据集 + 训练脚本 | 训练好的 checkpoints |
| **产出** | 权重文件 + loss 曲线 | 17 张诊断图表 + 结构化诊断报告 |
| **角色** | 引擎 (MLE) | 仪表盘 (DS) |
| **覆盖范围** | 单阶段独立训练 | 跨阶段交叉诊断 + 跨模态遗忘检测 |
| **学术深度** | 无文献引用 | 12 篇论文支撑的诊断方法论 |

---

## 📌 框架总览

```
┌──────────────────────────────────────────────────────────────────┐
│              LLM/VLM Training Diagnostic Framework               │
│                                                                  │
│  输入: 6 个阶段的 checkpoints + 文本 prompts + 图文测试对         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Module 1: Stage Goal Verification — 每个阶段达到目的了吗？       │
│  ├── LLM: 续写流畅度 / 指令遵循率 / 对齐精准度                   │
│  ├── 对齐税量化：DPO 前后逐维度能力变化追踪                       │
│  └── VLM: 视觉描述准确性 / 视觉问答能力 / 视觉指令遵循           │
│                                                                  │
│  Module 2: Capability Retention — 新阶段有没有破坏旧阶段成果？    │
│  ├── LLM→LLM: SFT 是否遗忘 pretrain 知识？                      │
│  ├── LLM→VLM: VLM 训练是否破坏纯文本能力？                      │
│  ├── VLM→VLM: VLM-SFT 是否破坏基础视觉描述？                    │
│  └── 归一化遗忘率 (Luo et al.) + VLM-CL 三类失败模式映射          │
│                                                                  │
│  Module 3: Pathology Detection — 模型有已知的病态行为吗？         │
│  ├── LLM: repetition / format overfitting / over-alignment /     │
│  │        mode collapse                                          │
│  ├── VLM: modality shortcut / description collapse /             │
│  │        visual hallucination / grounding failure                │
│  └── 幻觉来源归因（视觉编码器 / 投影层 / LLM 先验）               │
│                                                                  │
│  Module 4: Change Localization — 变化发生在模型哪部分？           │
│  ├── 参数漂移分析 (Attn Q/K/V vs FFN × shallow vs deep)          │
│  ├── 表征相似度 + CKA 预期范围校准                                │
│  └── VLM: 跨模态对齐度量 / 视觉信息流 / backbone 漂移             │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  输出: 17 张诊断图表 + PASS/WARN/FAIL 判定 + 优化建议             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📌 诊断覆盖的模型

### MiniMind LLM

MiniMind 是一个从零开始训练的超小语言模型，结构对齐 Qwen3 生态。
本项目使用的配置：`hidden_size=768, num_hidden_layers=8`，约 64M 参数。

| 阶段 | Checkpoint | 训练目标 |
|------|-----------|---------|
| Pretrain | `pretrain_768.pth` | 学习语言模式和世界知识 |
| SFT | `full_sft_768.pth` | 学会遵循指令 |
| GRPO | `grpo_768.pth` | 强化学习对齐人类偏好 |
| DPO | `dpo_768.pth` | 直接偏好优化 |

### MiniMind-V VLM

MiniMind-V 在 MiniMind LLM 基础上扩展视觉能力，添加 SigLIP2 视觉编码器 + MLP Projection（reshape 压缩 256→64 token），约 67M 参数。

| 阶段 | Checkpoint | 训练目标 |
|------|-----------|---------|
| VLM Pretrain | `vlm_pretrain_768.pth` | 视觉-语言对齐（仅训练 projection） |
| VLM SFT | `vlm_sft_768.pth` | 视觉指令遵循（全参数微调） |

<div align="center">

![VLM Structure](./images/LLM-structure.jpg)

</div>

---

## 📌 Quick Start

### 前置条件

```bash
# 克隆项目
git clone https://github.com/your-repo/minimind-diagnostic
cd minimind-diagnostic

# 安装依赖
pip install -r requirements.txt
```

### 准备 Checkpoints

将各阶段训练好的权重放入 `checkpoints/` 目录：

```
checkpoints/
├── pretrain_768.pth          # LLM pretrain
├── full_sft_768.pth          # LLM SFT
├── grpo_768.pth              # LLM GRPO
├── dpo_768.pth               # LLM DPO
├── vlm_pretrain_768.pth      # VLM pretrain (可选)
└── vlm_sft_768.pth           # VLM SFT (可选)
```

> 如果没有 VLM checkpoints，框架会自动跳过 VLM 相关诊断，仅运行 LLM 部分。

### 运行诊断

```bash
cd diagnostics

# 运行全部 4 个模块
python run_diagnostics.py

# 或指定模块运行
python run_diagnostics.py --module 1      # 仅 Module 1
python run_diagnostics.py --module 1 3    # Module 1 和 3
```

### 使用 Gemini API（推荐）

设置 Gemini API Key 可获得更准确的语义评估（标记为 `high confidence`）。
无 API 时框架自动回退到规则引擎评分（标记为 `medium confidence`）。

```bash
export GEMINI_API_KEY="your-api-key"
python run_diagnostics.py
```

### 产出文件

```
results/
├── raw/
│   ├── diagnostic_report.json    # 完整结构化结果
│   ├── stage_comparison.json     # M1 阶段对比
│   ├── retention_matrix.json     # M2 保留矩阵
│   ├── pathology_results.json    # M3 病理检测
│   └── drift_analysis.json       # M4 漂移分析
└── figures/                      # 17 张诊断图表
    ├── fig01 ~ fig17 (.png)

report/
├── diagnostic_report.md          # Markdown 诊断报告
└── literature_review.md          # 文献驱动叙事
```

---

## 📌 17 张诊断图表

### Module 1: Stage Goal Verification

| 图号 | 名称 | 回答的问题 |
|------|------|-----------|
| Fig 1 | Stage Goal Dashboard | 每个阶段达标了吗？ |
| Fig 2 | Alignment Precision Scatter | DPO 精准还是过度？ |
| Fig 3 | VLM Stage Comparison | VLM-SFT 比 Pretrain 好多少？ |
| Fig 4 | Behavior Transition | 行为模式怎么转变的？（5 类分类） |

<div align="center">
<img src="./results/figures/fig01_goal_dashboard.png" width="90%"/>
<p><i>Fig 1. Stage Goal Dashboard — 各阶段目标达成总览</i></p>
</div>

<div align="center">
<img src="./results/figures/fig02_alignment_scatter.png" width="70%"/>
<p><i>Fig 2. Alignment Precision — GRPO 达到安全拒绝 75%（PASS），DPO 退化至 25%（FAIL）</i></p>
</div>

<div align="center">
<img src="./results/figures/fig04_behavior_transition.png" width="70%"/>
<p><i>Fig 4. Behavior Transition — 从 Pretrain 续写到 SFT 指令遵循到 RLHF 安全拒绝的行为演化</i></p>
</div>

### Module 2: Capability Retention

| 图号 | 名称 | 回答的问题 |
|------|------|-----------|
| Fig 5 | Capability Retention Heatmap | 哪些能力被遗忘了？ |
| Fig 6 | Forgetting Rate Waterfall | 遗忘有多严重？（归一化） |
| Fig 7 | Cross-Modal Forgetting | VLM 破坏了文本能力吗？（含子维度分解） |

<div align="center">
<img src="./results/figures/fig05_retention_heatmap.png" width="80%"/>
<p><i>Fig 5. Capability Retention Heatmap — 事实知识在 SFT 后遗忘 50%（pretrain:2.0 → sft:1.0）</i></p>
</div>

<div align="center">
<img src="./results/figures/fig07_cross_modal_forgetting.png" width="80%"/>
<p><i>Fig 7. Cross-Modal Forgetting — VLM-SFT 文本能力反而优于纯 SFT（意外发现）</i></p>
</div>

### Module 3: Pathology Detection

| 图号 | 名称 | 回答的问题 |
|------|------|-----------|
| Fig 8 | Repetition Rate Trend | 重复在变好还是变差？ |
| Fig 9 | Paraphrase Consistency | SFT 真学会了还是死记？（含子维度分解） |
| Fig 10 | Mode Collapse Matrix | RLHF 后回答同质化了吗？ |
| Fig 11 | Visual Dependency Scores | VLM 真的在看图吗？ |
| Fig 12 | Hallucination & Grounding | VLM 幻觉率和接地能力 |

<div align="center">
<img src="./results/figures/fig09_paraphrase_consistency.png" width="80%"/>
<p><i>Fig 9. Paraphrase Consistency — SFT 后 consistency 仅 0.25（FAIL），含 topic/entity/structure 子维度</i></p>
</div>

<div align="center">
<img src="./results/figures/fig11_visual_dependency.png" width="70%"/>
<p><i>Fig 11. Visual Dependency — 视觉依赖 0.781（PASS），模型确实在看图</i></p>
</div>

### Module 4: Change Localization

| 图号 | 名称 | 回答的问题 |
|------|------|-----------|
| Fig 13 | Parameter Drift Heatmap | 哪些层/组件变化最大？ |
| Fig 14 | Representation Similarity | 变化在浅层还是深层？ |
| Fig 15 | Cross-Modal Alignment t-SNE | 视觉-文本对齐成功了吗？ |
| Fig 16 | Visual Information Flow | 视觉信息在哪层融合？ |
| Fig 17 | Backbone Drift Comparison | VLM 训练改了 LLM 的哪里？ |

<div align="center">
<img src="./results/figures/fig13_parameter_drift_heatmap.png" width="80%"/>
<p><i>Fig 13. Parameter Drift — Pretrain→SFT 变化最大（FFN 层 0.13），RLHF 阶段变化微小</i></p>
</div>

<div align="center">
<img src="./results/figures/fig14_representation_similarity.png" width="80%"/>
<p><i>Fig 14. Representation Similarity — Pretrain→SFT 深层 (layer 7) 相似度最低 0.888</i></p>
</div>

---

## 📌 诊断结果概览

以下为 MiniMind (768-dim, 8-layer) 完整 pipeline 的实际诊断结果。

### Module 1: Stage Goal Verification

| 阶段 | 指标 | 得分 | 状态 |
|------|------|------|------|
| Pretrain | 续写流畅度 (1-5) | 1.6 | ⚠️ WARN |
| Pretrain | 指令遵循率 | 0% | ❌ FAIL |
| SFT | 指令遵循率 | 12.5% | ❌ FAIL |
| SFT | 安全拒绝率 | 50% | ❌ FAIL |
| GRPO | 安全拒绝率 | 75% | ✅ PASS |
| DPO | 安全拒绝率 | 25% | ❌ FAIL |
| VLM-Pretrain | 视觉描述 (1-5) | 1.8 | ⚠️ WARN |
| VLM-SFT | 视觉问答 | QA:0.8 IF:0.4 | ⚠️ WARN |

**对齐税 (GRPO→DPO)**：平均 +0.22 [PASS]
- factual_knowledge: -0.33（无退化）
- instruction_quality: +0.67（退化明显）
- output_fluency: +0.33（轻度退化）

### Module 2: Capability Retention

| 能力维度 | Pretrain | SFT | GRPO | DPO | VLM-PT | VLM-SFT |
|----------|---------|-----|------|-----|--------|---------|
| 事实知识 | **2.0** ★ | 1.0 ⚠️ | 1.5 | 1.0 ⚠️ | - | - |
| 语言流畅 | 1.33 ★ | 2.0 | 2.0 | **3.0** | - | - |
| 指令遵循 | 2.33 | **2.0** ★ | 2.67 | 2.0 | - | - |
| 安全意识 | 1.0 | 3.0 | 3.0 | **1.5** ★ ⚠️ | - | - |
| 视觉描述 | - | - | - | - | **2.5** ★ | 2.5 |
| 视觉问答 | - | - | - | - | 1.0 | **5.0** ★ |

**关键遗忘**：
- ⚠️ 事实知识：pretrain→SFT 遗忘 50%（超过 15% 警戒线）
- ⚠️ 安全意识：dpo→pretrain 遗忘 33.3%
- ⚠️ 视觉问答：vlm_sft→vlm_pretrain 遗忘 80%（VLM-SFT 学到了 Pretrain 不具备的能力）

**VLM-CL 失败模式**：
- ⚠️ 跨模态特征漂移：paired_sim_delta = 0.022 < 0.1
- ⚠️ 共享模块干扰：shallow_drift = 0.112 > 2×deep = 0.008

### Module 3: Pathology Detection

| 病理 | 阶段 | 关键指标 | 状态 |
|------|------|---------|------|
| Token Repetition | pretrain/grpo/dpo | 4-gram rep ≤ 0.132 | ✅ PASS |
| Token Repetition | sft | 4-gram rep = 0.16 | ⚠️ WARN |
| Format Overfitting | sft | consistency = 0.25 | ❌ FAIL |
| Over-Alignment | grpo/dpo | false refusal ≤ 12.5% | ✅ PASS |
| Mode Collapse | grpo/dpo | response sim ≤ 0.085 | ✅ PASS |
| Modality Shortcut | vlm | visual dep = 0.781 | ✅ PASS |
| Description Collapse | vlm | cross-img sim = 0.289 | ✅ PASS |
| Visual Hallucination | vlm | hall rate = 87.5% | ❌ FAIL |
| Grounding Failure | vlm | grounding = 0.6 | ✅ PASS |

**幻觉来源归因**：主要来源为**视觉编码器表示质量**（visual_dependency=0.781 ≥ 0.2, projection_gain=0.215 ≥ 0.1）。建议考虑更强的视觉编码器或微调视觉编码器参数。

### Module 4: Change Localization

| 阶段转换 | 漂移模式 | 浅层均值 | 深层均值 |
|----------|---------|---------|---------|
| pretrain→sft | uniform | 0.0875 | 0.0964 |
| sft→grpo | uniform | 0.0012 | 0.0010 |
| grpo→dpo | uniform | 0.0012 | 0.0010 |
| sft→dpo | **shallow_dominant** | 0.0001 | 0.0000 |

**表征相似度**：pretrain→sft 最低 0.888 (layer 7)，RLHF 阶段 ≥ 0.994。

**因果推断**：轻微参数漂移，在可接受范围内 [PASS]。

---

## 📌 关键发现

### 发现 1：事实知识在 SFT 后严重遗忘

事实知识从 pretrain (2.0) → sft (1.0)，遗忘率 **50%**，远超 Luo et al. 建议的 15% 警戒线。这与大模型文献中 SFT 导致灾难性遗忘的实证一致（Luo et al., 2308.08747）。

**建议**：在 SFT 数据中混入预训练格式的知识保留数据。

### 发现 2：DPO 安全对齐退化

GRPO 将安全拒绝率提升至 75% (PASS)，但 DPO 后退化至 25% (FAIL)。这与 OGPSA (2602.07892) 中对齐税的持续学习遗忘理论一致。

**建议**：DPO 训练时增加安全数据比例，或使用梯度正交投影保留安全子空间。

### 发现 3：VLM 跨模态遗忘呈"共享模块干扰"模式

VLM 训练导致 LLM backbone 浅层漂移 0.112 远大于深层 0.008，符合 Laitinen & Imanov (2601.18699) 的低层 attention heads 脆弱性发现。

**建议**：冻结浅层参数或使用 O-LoRA 隔离模态相关参数。

### 发现 4：VLM 幻觉率高但确实在"看图"

视觉依赖 0.781 (PASS) 表明模型不是纯靠文本猜测，但幻觉率 87.5% 仍然很高。归因分析指向视觉编码器表示质量——SigLIP2-base 的表示可能不够忠实。

**建议**：使用更大的视觉编码器（如 SigLIP2-large）或微调视觉编码器。

### 发现 5：VLM-SFT 文本能力反超纯 SFT（意外）

VLM-SFT 纯文本质量 2.25 > SFT 纯文本质量 1.0，quality_drop 为负值。这在大模型文献中不常见，可能是小模型的独特现象——VLM 的多模态训练数据反而起到了数据增强作用。

---

## 📌 文献基础

本框架的诊断方法论和结果解释基于以下文献：

| 缩写 | 论文 | 应用于 |
|------|------|-------|
| MaP | Wang et al., 2025 | 评估不稳定性理论基础 |
| Luo | Luo et al. (2308.08747) | 归一化遗忘率公式 (M2) |
| Laitinen | Laitinen & Imanov (2601.18699) | 低层 attention heads 脆弱性 (M4) |
| OGPSA | (2602.07892) | 对齐税量化 (M1) |
| Yao | Yao et al. | 概念电路学习-遗忘权衡 (M4) |
| VLM-CL | Liu et al. (2508.04227) | VLM 持续学习三类失败模式 (M2) |
| Jing | Jing et al. (2505.01958) | 幻觉来源归因 (M3) |
| Grounding | (2509.10345) | 视觉接地分析 (M3) |
| CoSMo-RL | | 多目标 RL 安全约束 (M1) |
| Zucchet | Zucchet et al. | 知识涌现平台期 (M1) |

---

## 📌 项目结构

```
minimind-diagnostic/
├── diagnostics/                    ← 核心诊断代码
│   ├── module1_stage_goal.py       # M1: 阶段目标验证
│   ├── module2_retention.py        # M2: 能力保留测试
│   ├── module3_pathology.py        # M3: 病态行为检测
│   ├── module4_localization.py     # M4: 变化定位
│   ├── diagnostic_utils.py         # 工具函数 (推理/Gemini API/IO)
│   ├── run_diagnostics.py          # 主入口
│   ├── test_prompts.json           # 测试 prompts
│   └── utils/
│       ├── inference.py            # 模型推理
│       ├── scoring.py              # 评分逻辑
│       └── visualization.py        # 可视化
├── model/                          ← 模型定义
│   ├── model_minimind.py           # MiniMind LLM
│   ├── model_vlm.py                # MiniMind-V VLM
│   └── siglip2-base-p16-ve/        # SigLIP2 视觉编码器
├── trainer/                        ← 训练脚本
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   ├── train_grpo.py / train_dpo.py
│   ├── train_pretrain_vlm.py
│   └── train_sft_vlm.py
├── checkpoints/                    ← 各阶段权重
├── results/
│   ├── raw/                        # JSON 结构化结果
│   └── figures/                    # 17 张诊断图表
├── report/
│   ├── diagnostic_report.md        # Markdown 诊断报告
│   └── literature_review.md        # 文献驱动叙事
├── dataset/                        ← 训练数据
└── requirements.txt
```

---

## 📌 训练 Pipeline（来自原始项目）

本诊断框架分析的完整训练 pipeline 来自 MiniMind 与 MiniMind-V 项目：

### LLM 训练流程

```
Stage 1: Pretrain     → 学习语言模式和世界知识 → pretrain_768.pth
Stage 2: SFT          → 学会遵循指令          → full_sft_768.pth
Stage 3: GRPO         → 强化学习对齐偏好       → grpo_768.pth
Stage 4: DPO          → 直接偏好优化           → dpo_768.pth
```

### VLM 训练流程

```
Stage 5: VLM Pretrain → 视觉-语言对齐          → vlm_pretrain_768.pth
                        (冻结 LLM，仅训练 projection)
Stage 6: VLM SFT      → 视觉指令遵循           → vlm_sft_768.pth
                        (全参数微调)
```

VLM 的架构在 LLM 基础上添加了 SigLIP2 视觉编码器 + MLP Projection：
- SigLIP2 输出 256×768 patch features
- Reshape 压缩 (256×768 → 64×3072)
- 2-layer MLP 投影至 LLM 维度 (64×768)
- 64 个视觉 token 替换 `<|image_pad|>` 占位符的 embedding

如需从零训练模型，请参考：
- LLM：[MiniMind 项目](https://github.com/jingyaogong/minimind)
- VLM：[MiniMind-V 项目](https://github.com/jingyaogong/minimind-v)

---

## 📌 训练问题 → 诊断模块映射

框架覆盖 **15 个训练中的真实 failure mode**：

### Pretrain 阶段

| 问题 | 症状 | 主诊断 | 文献 |
|------|------|--------|------|
| Token Repetition | 输出陷入循环重复 | M3: n-gram 重复率 | — |
| Degeneration | 退化为高频词堆砌 | M1: 续写流畅度 | Zucchet: 知识涌现平台期 |

### SFT 阶段

| 问题 | 症状 | 主诊断 | 文献 |
|------|------|--------|------|
| Format Overfitting | 换一种问法就不会了 | M3: paraphrase consistency | — |
| Knowledge Collapse | 只会对话，忘了知识 | M2: retention matrix | Luo: SFT 遗忘实证 |

### GRPO / DPO 阶段

| 问题 | 症状 | 主诊断 | 文献 |
|------|------|--------|------|
| Over-Alignment | 对无害请求也拒绝 | M1: false refusal rate | OGPSA: 对齐税 |
| Mode Collapse | 所有回答同一模板 | M3: cross-response sim | — |
| Alignment Tax | 安全性↑但质量↓ | M2: retention matrix | OGPSA: SimpleQA 坍缩 |

### VLM Pretrain 阶段

| 问题 | 症状 | 主诊断 | 文献 |
|------|------|--------|------|
| Projection Failure | 视觉未映射到语言空间 | M4: paired cosine sim | VLM-CL: 跨模态漂移 |
| Description Collapse | 所有图片输出相同描述 | M3: cross-image sim | Jing: 投影层信息损失 |
| Modality Shortcut | 不看图纯靠文本猜 | M3: visual ablation | Grounding: 语言先验 |

### VLM SFT 阶段

| 问题 | 症状 | 主诊断 | 文献 |
|------|------|--------|------|
| Visual Hallucination | 描述图中不存在的物体 | M3: hallucination check | Jing: 多组件归因 |
| Grounding Failure | 回答与图片不对应 | M3: correct vs distractor | Grounding: 空间精度 |
| Language Forgetting | VLM 破坏了文本能力 | M2: cross-modal forgetting | VLM-CL: 三类失败模式 |

---

## 📌 技术细节

### 评分机制

- **Gemini API 模式**（`judge_backend: gemini, confidence: high`）：使用 Gemini 2.5 Flash 进行语义评估
- **离线规则模式**（`judge_backend: offline_rules, confidence: medium`）：多信号规则引擎自动回退
  - 文本质量：分解为 relevance（关键词命中率）、completeness（长度+结构）、repetition_penalty（去重惩罚）三子维度
  - 语义一致性：分解为 topic_match、entity_match、structure_match 三子维度
  - 遗忘率：采用 Luo et al. 归一化公式，裁剪至 [-1.0, 1.0]

### 参数漂移分析

按功能模块（Attn Q/K/V、FFN、Embedding）× 深度（Shallow vs Deep）双重分组，对齐 Laitinen 的低层 attention heads 脆弱性发现。

### VLM-CL 失败模式映射

检测到跨模态遗忘后，自动映射到 VLM-CL 综述的三类失败模式：
1. **跨模态特征漂移** — paired cosine sim 提升不足
2. **共享模块干扰** — backbone 浅层漂移显著大于深层
3. **零样本能力侵蚀** — text-only 评分下降

### 幻觉来源归因

检测到幻觉后，根据 Jing et al. 三组件分析归因至：
- 视觉编码器表示质量（visual_dep ≥ 0.2 且 projection_gain ≥ 0.1 时）
- 投影层失败（projection_gain < 0.1 时）
- LLM 语言先验主导（visual_dep < 0.2 时）

---

## 📌 环境要求

| 项目 | 要求 |
|------|------|
| Python | ≥ 3.10 |
| GPU | 任何 CUDA GPU（推理模式，~2GB 显存） |
| 核心依赖 | PyTorch, Transformers, scikit-learn, matplotlib, numpy, PIL |
| 可选 | google-genai（Gemini API 评分） |
| 运行时间 | ~35 min（4 module 全跑，含 VLM） |

---

## 📌 License

本项目基于 [Apache 2.0](./LICENSE) 协议开源。

---

## 📌 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) — LLM 训练教程与模型代码
- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — VLM 训练教程与视觉扩展
- [SigLIP2](https://huggingface.co/jingyaogong/siglip2-base-p16-ve) — 视觉编码器
- 12 篇支撑诊断方法论的学术论文（详见[文献综述](./report/literature_review.md)）
