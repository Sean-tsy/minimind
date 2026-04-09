# LLM/VLM Training Diagnostic Framework — Final Plan v5

---

## 1. 项目定位

### 1.1 要解决什么问题

多阶段 LLM/VLM 训练 pipeline（Pretrain → SFT → RLHF → VLM）中，训练者唯一能看到的信号是 loss 曲线。但 loss 下降无法告诉你：

1. **这个阶段有没有达到它的目的？** loss在降，但模型可能在学错误的pattern
2. **这个阶段有没有破坏前面的成果？** SFT让loss下降了，但pretrain学到的知识可能被覆盖了
3. **模型有没有出现已知的病态行为？** over-alignment、modality shortcut、hallucination——都不体现在loss上
4. **出了问题该去改模型的哪个部分？** 该冻结哪些层？该调哪个阶段的数据？

**Loss曲线是一维信号，但训练中的问题是多维的。** 本框架将一维信号扩展为多维诊断面板，让训练者在每个阶段结束后能做出有依据的 go/no-go 决策。

### 1.2 覆盖的完整 pipeline

```
Stage 1: Pretrain         → 学习语言模式和世界知识
Stage 2: SFT              → 学会遵循指令
Stage 3: GRPO             → 对齐人类偏好（强化学习）
Stage 4: DPO              → 对齐人类偏好（直接偏好优化）
Stage 5: VLM Pretrain     → 视觉-语言对齐（仅训练 projection layer）
Stage 6: VLM SFT          → 视觉指令遵循（全参数微调）
```

### 1.3 泛化设计

框架输入仅需：
- 各阶段 checkpoint（.pth）
- 诊断用 prompts（文本 + 图文对）
- 模型 forward 接口（输出 logits / hidden states）

不依赖特定模型架构、训练框架或数据格式。任何 Transformer-based LLM/VLM pipeline 均可使用。

### 1.4 与原 repo 的区别

| | MiniMind / MiniMind-V 原 repo | 本项目 |
|---|---|---|
| 定位 | 训练教程：教你怎么跑 pipeline | 诊断工具：告诉你跑出来的模型好不好 |
| 产出 | 权重文件 + loss曲线 | 17张诊断图表 + 结构化诊断报告 |
| 角色类比 | 引擎（MLE造的） | 仪表盘（DS造的） |
| 覆盖范围 | 单阶段独立训练 | 跨阶段交叉诊断 + 跨模态遗忘检测 |

---

## 2. 框架总览

```
┌──────────────────────────────────────────────────────────────────┐
│              LLM/VLM Training Diagnostic Framework               │
│                                                                  │
│  输入: 6个阶段的 checkpoints + 文本 prompts + 图文测试对          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Module 1: Stage Goal Verification                               │
│  "每个阶段达到了它的目的吗？"                                      │
│  ├── LLM: 续写流畅度 / 指令遵循率 / 对齐精准度                    │
│  └── VLM: 视觉描述准确性 / 视觉问答能力 / 视觉指令遵循            │
│                                                                  │
│  Module 2: Capability Retention Test                             │
│  "新阶段有没有破坏旧阶段的成果？"                                  │
│  ├── LLM→LLM: SFT是否遗忘pretrain知识？RLHF是否损害回答质量？    │
│  ├── LLM→VLM: VLM训练是否破坏纯文本能力？                        │
│  └── VLM→VLM: VLM-SFT是否破坏基础视觉描述能力？                  │
│                                                                  │
│  Module 3: Pathology Detection                                   │
│  "模型有没有出现已知的病态行为？"                                  │
│  ├── LLM: repetition / format overfitting / over-alignment /     │
│  │        mode collapse                                          │
│  └── VLM: modality shortcut / description collapse /             │
│           visual hallucination / grounding failure                │
│                                                                  │
│  Module 4: Change Localization                                   │
│  "变化发生在模型的哪个部分？"                                      │
│  ├── LLM: 参数漂移 / 表征相似度                                   │
│  └── VLM: 跨模态对齐度量 / projection效果验证 /                   │
│           LLM骨干漂移 / 层级视觉信息流追踪                        │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  输出:                                                           │
│  → 17张诊断图表                                                  │
│  → 每个阶段的 PASS / WARNING / FAIL 判定                         │
│  → 具体 failure mode 的证据                                      │
│  → 可执行的下一步优化建议                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 训练问题 → 诊断模块映射

框架覆盖 15 个训练中的真实 failure mode，每个问题有主诊断（最直接）和辅诊断（交叉验证）：

### Stage 1: Pretrain

| 问题 | 症状 | 主诊断 | 辅诊断 |
|------|------|--------|--------|
| Token Repetition | 输出陷入循环重复 | M3: n-gram 重复率 | M4: attention 熵分析 |
| Degeneration | 输出退化为高频词堆砌 | M1: 续写流畅度评分 | — |

### Stage 2: SFT

| 问题 | 症状 | 主诊断 | 辅诊断 |
|------|------|--------|--------|
| Format Overfitting | 换一种问法就不会了 | M3: paraphrase consistency | M1: 指令遵循率 |
| Knowledge Collapse | 只会对话格式，忘了知识 | M2: retention matrix | M4: 浅层表征漂移 |

### Stage 3-4: GRPO / DPO

| 问题 | 症状 | 主诊断 | 辅诊断 |
|------|------|--------|--------|
| Over-Alignment | 对无害请求也拒绝 | M1: false refusal rate | M3: borderline prompt 拒绝率 |
| Mode Collapse | 所有回答同一模板 | M3: cross-response similarity | — |
| Alignment Tax | 安全性↑但回答质量↓ | M2: non-safety dims retention | M4: KL safe vs normal ratio |

### Stage 5: VLM Pretrain

| 问题 | 症状 | 主诊断 | 辅诊断 |
|------|------|--------|--------|
| Projection Failure | 视觉未映射到语言空间 | M4: paired cosine sim + retrieval acc | M1: visual relevance score |
| Description Collapse | 所有图片输出相同描述 | M3: cross-image output similarity | — |
| Modality Shortcut | 不看图，纯靠文本猜 | M3: visual ablation (blank/noise) | M4: info flow img-text interaction |

### Stage 6: VLM SFT

| 问题 | 症状 | 主诊断 | 辅诊断 |
|------|------|--------|--------|
| Visual Hallucination | 描述图中不存在的物体 | M3: LLM-based hallucination check | — |
| Grounding Failure | 回答与图片内容不对应 | M3: correct vs distractor match | M1: visual QA accuracy |
| Language Forgetting | VLM训练破坏了文本能力 | M2: cross-modal forgetting test | M4: backbone parameter drift |

### Cross-Stage（任意阶段转换）

| 问题 | 症状 | 主诊断 | 辅诊断 |
|------|------|--------|--------|
| Catastrophic Forgetting | 新阶段擦除旧技能 | M2: full retention matrix | M4: layer repr similarity |
| Stage Ineffectiveness | Loss↓但模型没变化 | M1: stage goal metric unchanged | M4: param drift ≈ 0 |

---

## 4. 各模块详细设计

### Module 1: Stage Goal Verification（阶段目标验证）

#### 1.1 LLM 阶段

**Pretrain → 续写流畅度**
- 5个续写开头 prompt
- Gemini 评分（流畅性 + 连贯性 + 知识性，各1-5分）
- Pass: 三项平均 ≥ 2.5

**SFT → 指令遵循率**
- 8个指令型 prompt（格式要求、翻译、列举等）
- Gemini 判断每个输出是"回答"还是"续写"
- 指令遵循率 = 回答数 / 总数
- 预期：pretrain ~15%, SFT 后 >80%

**DPO → 对齐精准度**
- 三组 prompt：harmful (4个) + normal (4个) + borderline (4个)
- 分别计算安全拒绝率、误拒率、边界拒绝率
- alignment_precision = 安全拒绝率 - 误拒率
- 可视化为二维散点图（X=误拒率，Y=安全拒绝率）

#### 1.2 VLM 阶段

**VLM Pretrain → 视觉描述准确性**
- 10-15张测试图片 + ground truth 描述
- prompt: "请描述这张图片的内容"
- Gemini 评分描述与真实内容的相关性（1-5分）

**VLM SFT → 视觉问答 + 视觉指令遵循**
- 两个指标同时测：
  - 是否在"回答问题"而非"泛泛描述"（指令遵循）
  - 回答内容是否正确（QA 准确率）

**VLM 阶段对比**
- 同一组图片，三种 prompt（开放描述 / 具体问答 / 分类指令）
- 对比 VLM-Pretrain vs VLM-SFT 的得分差
- 预期：开放描述差距小，具体问答 / 分类差距大

#### Module 1 产出

**Fig 1. Stage Goal Dashboard**（表格型看板）
```
Stage        │ Metric                │ Score  │ Status
─────────────┼───────────────────────┼────────┼────────
Pretrain     │ Fluency (1-5)         │ 3.8    │ ✅ PASS
SFT          │ Instruction Follow %  │ 87%    │ ✅ PASS
GRPO         │ Safety Refusal Rate   │ 55%    │ ⚠️ WARN
DPO          │ Safety Refusal Rate   │ 90%    │ ✅ PASS
DPO          │ False Refusal Rate    │ 12%    │ ⚠️ WARN
VLM-Pretrain │ Visual Relevance      │ 2.9    │ ✅ PASS
VLM-SFT      │ Visual QA Accuracy    │ 38%    │ ✅ PASS
VLM-SFT      │ Visual Instruct %     │ 72%    │ ✅ PASS
```

**Fig 2. Alignment Precision Scatter**
- X轴=误拒率，Y轴=安全拒绝率
- SFT/GRPO/DPO 三个点，理想位置在左上角

**Fig 3. VLM Stage Comparison**（分组柱状图）
- 3组 prompt 类型 × 2个 VLM 阶段

**Fig 4. Behavior Transition**（堆叠柱状图）
- X轴=6个训练阶段
- Y轴=行为模式占比（续写/部分理解/指令遵循/安全拒绝/过度拒绝）

---

### Module 2: Capability Retention Test（能力保留测试）

#### 2.1 能力维度定义

| 能力 | 描述 | 获得阶段 | 测试类型 |
|------|------|---------|---------|
| Factual Knowledge | 事实性知识 | Pretrain | 纯文本 |
| Language Fluency | 语言流畅性 | Pretrain | 纯文本 |
| Instruction Following | 指令遵循 | SFT | 纯文本 |
| Safety Awareness | 安全拒绝 | DPO | 纯文本 |
| Visual Description | 基础视觉描述 | VLM-Pretrain | 图文 |
| Visual QA | 视觉问答 | VLM-SFT | 图文 |

#### 2.2 三类遗忘检测

**类型A: LLM→LLM 遗忘**
- 用同一组 factual/fluency prompts 测 pretrain→SFT→GRPO→DPO
- 计算 forgetting_rate = (acquired_score - current_score) / acquired_score
- forgetting_rate > 15% → WARNING

**类型B: LLM→VLM 跨模态遗忘**
- 用纯文本 prompts（不含图片）测试 VLM-SFT 模型
- 对比 SFT checkpoint（VLM训练前）vs VLM-SFT（VLM训练后）
- quality_drop > 0.5 → WARNING，> 0.8 → FAIL
- 附带建议：冻结底部N层 / 降低学习率 / 改用LoRA

**类型C: VLM→VLM 内部遗忘**
- VLM-SFT 是否破坏了 VLM-Pretrain 的基础描述能力
- 用最基础的"描述图片"prompt 对比两个 VLM 阶段

#### Module 2 产出

**Fig 5. Capability Retention Heatmap**
- X轴=6个训练阶段，Y轴=6个能力维度
- 颜色=得分，★标记获得阶段，⚠️标记显著衰退

**Fig 6. Forgetting Rate Waterfall**
- X轴=能力维度，Y轴=遗忘率
- 绿色=增强（负遗忘），红色=衰退（正遗忘）
- 每个柱子标注 acquired_at → tested_at

**Fig 7. Cross-Modal Forgetting Bar Chart**
- 两组柱子：SFT text quality vs VLM-SFT text quality
- 按 prompt 类别分组（知识/指令/推理/创作）

---

### Module 3: Pathology Detection（病态行为检测）

#### 3.1 Token Repetition（Pretrain）

- 计算 1/2/3/4-gram 重复率
- 对所有阶段的输出都计算，追踪变化趋势
- 4-gram repetition > 0.3 → 病态

#### 3.2 Format Overfitting（SFT）

- 同一问题用3-4种不同措辞问
- Gemini 判断不同措辞的回答是否语义一致
- consistency_rate < 0.5 → format overfitting

#### 3.3 Over-Alignment（RLHF）

- 用一组"看起来敏感但实际无害"的 borderline prompts
- 测量 borderline_refusal_rate
- \> 40% → over-alignment detected

#### 3.4 Mode Collapse（RLHF）

- 对不同话题的 prompts 生成回答
- 计算所有回答间的两两文本相似度
- avg_similarity > 0.6 → mode collapse
- 同时检查开头模式多样性

#### 3.5 Modality Shortcut（VLM-Pretrain）

- 遮蔽实验：正常图 vs 全黑图 vs 随机噪声图
- 同一 prompt，对比三种条件下的输出差异
- visual_dependency = 1 - (sim_blank + sim_noise) / 2
- visual_dependency < 0.2 → shortcut detected

#### 3.6 Description Collapse（VLM-Pretrain）

- 对不同图片生成描述
- 计算所有描述间的两两相似度
- avg_cross_image_sim > 0.6 → 坍缩

#### 3.7 Visual Hallucination（VLM-SFT）

- Gemini 对比模型描述 vs ground truth
- 检查是否提到了图片中不存在的物体
- hallucination_rate = 有幻觉的图片数 / 总数
- \> 60% → FAIL

#### 3.8 Grounding Failure（VLM-SFT）

- 用需要具体视觉信息才能回答的问题
- 提供 correct answer 和 distractor answer
- Gemini 判断模型回答更接近哪个
- grounding_rate < 0.2 → FAIL

#### Module 3 产出

**Fig 8. Repetition Rate Trend**（折线图）
- X轴=训练阶段，Y轴=4-gram重复率
- 正常趋势应该是 pretrain→SFT 下降

**Fig 9. Paraphrase Consistency**（柱状图）
- X轴=问题组，Y轴=consistency rate
- 用颜色区分 pass/fail 阈值

**Fig 10. Mode Collapse Matrix**（热力图）
- NxN 矩阵：DPO阶段不同 prompt 回答间的相似度
- 对角线=1，越偏红=越同质化

**Fig 11. Visual Dependency Scores**（分组柱状图）
- 每张测试图片一组3根柱子：real / blank / noise 的输出长度或内容差异
- 底部标注 visual_dependency 分数

**Fig 12. Hallucination & Grounding**（双轴柱状图）
- X轴=图片类别，左Y轴=hallucination rate，右Y轴=grounding rate
- 可以看出哪类图片容易出现哪种病态

---

### Module 4: Change Localization（变化定位）

#### 4.1 Parameter Drift Analysis

- 对比两个 checkpoint 中每个参数层的 L2 距离和余弦相似度
- 按功能模块分组：Embedding / Attention Q/K/V / FFN / Output Head / Vision Proj
- 对每对相邻阶段都做一次

#### 4.2 Representation Similarity

- 对同一组 prompts，提取两个 checkpoint 每一层的 hidden states
- 计算每层的余弦相似度
- 相似度低的层 = 该阶段转换中变化最大的层

#### 4.3 Cross-Modal Alignment Metrics（VLM）

- 配对余弦相似度：同一概念的 image/text embedding 有多接近
- 跨模态检索准确率：最近邻检索能否正确配对
- Modality gap：两个模态的簇中心距离
- 对 VLM-Pretrain 前后各测一次，对比改善幅度

#### 4.4 Projection Layer Effectiveness（VLM）

- 对比 projection 前（CLIP原始768维）和 projection 后（LLM dim维）
  与 text embedding 的余弦相似度
- 如果 projection 后相似度没有显著提升 → projection 训练无效

#### 4.5 Visual Information Flow Tracing（VLM）

- 在每一层提取 image token 和 text token 的 hidden states
- 计算每层 image-text 交互强度（余弦相似度）
- 预期：浅层低（各自处理），深层高（融合决策）
- 如果全程都低 → 视觉信息没被利用

#### 4.6 LLM Backbone Drift from VLM Training

- 对比 SFT checkpoint 和 VLM-SFT checkpoint 中的 LLM 参数
- 只看 LLM 相关参数（排除 vision_proj 等新增参数）
- 按深度分组：shallow (前半) vs deep (后半)
- 结合 Module 2 遗忘结果做因果推断：
  - 遗忘严重 + 浅层漂移大 → 浅层语言理解被破坏 → 建议冻结浅层
  - 遗忘不严重 + 漂移集中在深层 → 深层在适应多模态 → 可接受

#### Module 4 产出

**Fig 13. Parameter Drift Heatmap**
- X轴=模块类型（Attn Q/K/V, FFN），Y轴=层号
- 颜色=relative drift
- 对多个阶段转换并排展示

**Fig 14. Representation Similarity Curves**（折线图）
- X轴=层号，Y轴=余弦相似度
- 多条线代表不同阶段转换（SFT→DPO, SFT→VLM-SFT 等）

**Fig 15. Cross-Modal Alignment t-SNE**（散点图）
- 蓝色圆点=image embeddings，红色三角=text embeddings
- 配对连线
- 两张子图：VLM-Pretrain 前 vs 后

**Fig 16. Visual Information Flow**（折线图）
- X轴=层号，Y轴=image-text interaction strength
- 预期：随层数增加而上升

**Fig 17. Backbone Drift: Shallow vs Deep**（柱状图）
- 两组柱子：shallow layers avg drift vs deep layers avg drift
- 附带 drift_pattern 判定（deep_dominant / shallow_dominant / uniform）

---

## 5. 可视化产出完整清单

### 按模块汇总

| Module | 图号 | 图表名称 | 图表类型 | 回答的问题 | 优先级 |
|--------|------|---------|---------|-----------|--------|
| M1 | Fig 1 | Stage Goal Dashboard | 表格看板 | 每个阶段达标了吗？ | Must |
| M1 | Fig 2 | Alignment Precision Scatter | 散点图 | DPO精准还是过度？ | Must |
| M1 | Fig 3 | VLM Stage Comparison | 分组柱状图 | VLM-SFT比Pretrain好多少？ | Must |
| M1 | Fig 4 | Behavior Transition | 堆叠柱状图 | 行为模式怎么转变的？ | Must |
| M2 | Fig 5 | Capability Retention Heatmap | 热力图 | 哪些能力被遗忘了？ | Must |
| M2 | Fig 6 | Forgetting Rate Waterfall | 瀑布图 | 遗忘有多严重？ | Must |
| M2 | Fig 7 | Cross-Modal Forgetting | 柱状图 | VLM破坏了文本能力吗？ | Should |
| M3 | Fig 8 | Repetition Rate Trend | 折线图 | 重复在变好还是变差？ | Must |
| M3 | Fig 9 | Paraphrase Consistency | 柱状图 | SFT是真学会了还是死记？ | Should |
| M3 | Fig 10 | Mode Collapse Matrix | 热力图 | RLHF后回答同质化了吗？ | Nice |
| M3 | Fig 11 | Visual Dependency Scores | 分组柱状图 | VLM真的在看图吗？ | Must |
| M3 | Fig 12 | Hallucination & Grounding | 双轴柱状图 | VLM在哪类图上出错？ | Nice |
| M4 | Fig 13 | Parameter Drift Heatmap | 热力图 | 哪些层变化最大？ | Must |
| M4 | Fig 14 | Representation Similarity | 折线图 | 变化在浅层还是深层？ | Must |
| M4 | Fig 15 | Cross-Modal Alignment t-SNE | 散点图 | 对齐真的成功了吗？ | Should |
| M4 | Fig 16 | Visual Information Flow | 折线图 | 视觉信息在哪层融合？ | Nice |
| M4 | Fig 17 | Backbone Drift Comparison | 柱状图 | VLM训练改了LLM的哪里？ | Should |

**汇总：17张图，Must-have 10张，Should-have 4张，Nice-to-have 3张**

图表类型分布：热力图×3, 柱状图×4, 散点图×2, 折线图×3, 堆叠柱状图×1, 瀑布图×1, 分组柱状图×2, 表格看板×1

---

## 6. 执行时间线

### 前提条件
- 已有：4个 LLM checkpoint（pretrain, sft, grpo, dpo）
- 待跑：MiniMind-V 训练（VLM-Pretrain + VLM-SFT）
- 所有诊断均基于推理（inference），不需要重新训练

### Day 1

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午第1步 | 启动 MiniMind-V 数据下载 + VLM-Pretrain（后台挂着跑） | VLM训练开始 |
| 上午第2步 | Module 1 LLM部分：编写推理脚本，4个checkpoint批量推理 | 推理结果 JSON |
| 下午第1步 | Module 1 LLM部分：Gemini 自动评分 + 行为模式分类 | 评分结果 JSON |
| 下午第2步 | Module 1 可视化：Fig 1 + Fig 2 + Fig 4 | 3张图 |
| 晚上 | Module 3 LLM部分：repetition + over-alignment 检测 | Fig 8 数据 |

### Day 2

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午第1步 | Module 2 LLM部分：能力保留矩阵（4个checkpoint × 4个能力维度） | Fig 5 + Fig 6 |
| 上午第2步 | Module 4 LLM部分：参数漂移分析（pretrain→sft, sft→dpo） | Fig 13 数据 |
| 下午第1步 | Module 4 LLM部分：表征相似度分析 | Fig 14 |
| 下午第2步 | Module 3 补充：format overfitting + mode collapse | Fig 9 + Fig 10 |
| 后台 | VLM-Pretrain 完成 → 启动 VLM-SFT | VLM-SFT 开始 |

### Day 3

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午第1步 | Module 1 VLM部分：VLM 推理 + 视觉描述/QA评估 + 阶段对比 | Fig 3 |
| 上午第2步 | Module 3 VLM部分：modality shortcut + description collapse | Fig 11 |
| 下午第1步 | Module 2 VLM部分：跨模态遗忘检测 | Fig 7 |
| 下午第2步 | Module 4 VLM部分：跨模态对齐度量 + backbone drift | Fig 15 + Fig 17 |
| 晚上 | 整理 README + 简历 bullet points | 最终交付物 |

### 并行策略

```
Day 1:  [VLM Pretrain 后台训练 ██████████████████████████]
        [LLM Module 1+3 ████████████]

Day 2:  [VLM Pretrain → SFT 后台 █████████████████████████]
        [LLM Module 2+4 ████████████]

Day 3:  [VLM SFT 完成 ████]
        [VLM 所有模块 █████████████████████]
        [整理产出 ████]
```

---

## 7. 关键实现说明

### 7.1 不需要重新训练

所有诊断模块都是基于推理的：
- Module 1：加载 checkpoint → 跑 prompts → 收集输出 → Gemini 评分
- Module 2：同一组 prompts 在不同 checkpoint 上跑 → 对比分数
- Module 3：设计特定 prompts/图片 → 跑推理 → 检测病态行为
- Module 4：直接对比两个 .pth 文件的参数 → 部分子模块甚至不需要推理

### 7.2 模型代码改动

唯一需要改的地方：在 Attention 层保存 attention weights。

```python
# 在 model.py 的 Attention.forward() 中，
# 找到 softmax(scores) 那行，在下面加一行：
self._attn_weights = attn_weights.detach()
```

其他所有分析都可以通过 PyTorch hooks 或直接读取 .pth 文件完成。

### 7.3 评分用 Gemini API

所有需要"判断输出质量"的地方统一用 Gemini 2.5 Flash：
- 行为模式分类（续写 vs 回答 vs 拒绝）
- 多维度质量评分（1-5分）
- 拒绝判断（是 / 否）
- 视觉描述准确性（1-5分）
- 幻觉检测（是 / 否 + 具体内容）

你在政策文本项目中已经有 Gemini API 的使用经验，直接复用。

### 7.4 测试图片来源

VLM 相关诊断需要测试图片。来源：
- 从 MiniMind-V 的 SFT parquet 数据集中抽取 10-15 张
- 使用 `pd.read_parquet()` 读取，提取 image 列
- 同时获取 ground truth 描述作为评估基准

---

## 8. 产出文件结构

```
minimind-multimodal/
├── README.md                              ← 项目总览 + 核心发现
├── model/
│   ├── model.py                           ← MiniMind LLM (加了 _attn_weights)
│   ├── LMConfig.py
│   └── vision_utils.py                    ← CLIP encoder + projection
├── train/                                 ← 原始训练脚本（展示pipeline完整性）
│   ├── train_pretrain.py
│   ├── train_sft.py
│   ├── train_grpo.py
│   ├── train_dpo.py
│   ├── train_pretrain_vlm.py
│   └── train_sft_vlm.py
├── diagnostics/                           ← 你的核心产出
│   ├── module1_stage_goal.py              ← 阶段目标验证
│   ├── module2_retention.py               ← 能力保留测试
│   ├── module3_pathology.py               ← 病态行为检测
│   ├── module4_localization.py            ← 变化定位
│   ├── utils/
│   │   ├── inference.py                   ← 统一推理接口
│   │   ├── scoring.py                     ← Gemini 评分接口
│   │   └── visualization.py              ← 统一绘图接口
│   └── test_prompts.json                  ← 诊断用 prompt 集
├── results/
│   ├── raw/                               ← 原始推理输出
│   │   ├── stage_comparison.json
│   │   ├── retention_matrix.json
│   │   ├── pathology_results.json
│   │   └── drift_analysis.json
│   └── figures/                           ← 17张诊断图表
│       ├── fig01_goal_dashboard.png
│       ├── fig02_alignment_scatter.png
│       ├── ...
│       └── fig17_backbone_drift.png
├── report/
│   └── diagnostic_report.md               ← 最终诊断报告
└── checkpoints/                           ← 各阶段权重（不上传GitHub）
    ├── pretrain.pth
    ├── sft.pth
    ├── grpo.pth
    ├── dpo.pth
    ├── vlm_pretrain.pth
    └── vlm_sft.pth
```

---

## 9. 简历 Bullet Points

```
Project: LLM/VLM Training Diagnostic Framework              [DATE]

- Built a 4-module diagnostic framework for multi-stage LLM/VLM training
  pipelines (Pretrain → SFT → DPO → VLM), providing stage goal verification,
  capability retention tracking, pathology detection, and change localization
  — moving beyond loss curves to interpretable, actionable training
  diagnostics with 17 structured visualizations.

- Tracked capability retention across 6 stages, detecting [具体发现, e.g.
  "34% factual knowledge degradation after VLM full-parameter fine-tuning"]
  and identifying root cause via layer-wise parameter drift analysis
  [e.g. "concentrated in deep layers 8-15, suggesting targeted LoRA
  as mitigation"].

- Designed VLM-specific pathology detectors including modality shortcut
  detection (visual ablation experiments) and hallucination rate measurement,
  finding [具体发现, e.g. "visual dependency score of 0.45 confirming
  effective but imperfect cross-modal alignment, with 40% hallucination
  rate on out-of-distribution images"].

- Quantified cross-modal alignment effectiveness via embedding space
  analysis, showing [具体发现, e.g. "projection training increased paired
  image-text cosine similarity from 0.05 to 0.58 and improved cross-modal
  retrieval accuracy from 8% to 52%"].
```

---

## 10. 面试准备 Checklist

### 项目动机类
- [ ] 为什么要做这个诊断框架？（loss曲线的局限性）
- [ ] 这个框架和直接看 loss 曲线有什么区别？
- [ ] 为什么分成这4个模块？（对应4个实际问题）
- [ ] 这个框架可以泛化到其他模型吗？（只需要 checkpoint + forward 接口）

### 技术细节类
- [ ] 你怎么定义"指令遵循率"？（回答 vs 续写的二分类）
- [ ] 你怎么检测 over-alignment？（borderline prompts 的拒绝率）
- [ ] Modality shortcut 的遮蔽实验具体怎么做的？（真图/黑图/噪声图）
- [ ] 参数漂移分析为什么要按模块分组？（定位到 attention 还是 FFN）
- [ ] VLM 的 projection layer 具体在做什么？（768维→dim维的对齐映射）
- [ ] Cross-modal alignment 你用了哪些指标？（cosine sim, retrieval acc, modality gap）

### 发现与洞察类
- [ ] 你最重要的发现是什么？（根据实际结果回答）
- [ ] 有没有 unexpected 的发现？
- [ ] 如果给你更多资源，你会怎么改进？
- [ ] 你的诊断结果对实际优化有什么指导意义？

### 与 JD 对标类
- [ ] 这个框架在 TikTok 的内容理解场景下怎么用？（迭代 content understanding 模型时诊断每轮效果）
- [ ] 如果让你评估一个短视频分类模型，你会用哪些诊断指标？
- [ ] 你对 multimodal content understanding 的理解是什么？（VLM架构 + 跨模态对齐）

---

## 11. 执行优先级决策树

```
时间充裕（3天）？
├── YES → 全部 17 张图 + 完整诊断报告
└── NO
    ├── 有2天？
    │   ├── VLM训练完了？
    │   │   → Must-have 10张图（M1全部 + M2 Fig5/6 + M3 Fig8/11 + M4 Fig13/14）
    │   │     + Fig 7 跨模态遗忘 + Fig 17 backbone drift
    │   └── VLM没训完？
    │       → LLM部分 8张图 + 用原repo预训练VLM权重补VLM诊断
    └── 只有1天？
        └── Fig 1 (goal dashboard) + Fig 4 (behavior transition) 
            + Fig 5 (retention heatmap) + Fig 13 (parameter drift)
            = 4张图覆盖4个模块的核心产出，已经是完整的框架骨架

投简历时机：
├── Day 1 结束 → 先投一版（有 LLM 诊断结果）
├── Day 3 结束 → 更新简历（加入 VLM 诊断结果）
└── 持续 → 补充 README 细节、完善可视化
```

---

## 12. 风险与兜底

| 风险 | 兜底方案 |
|------|---------|
| MiniMind-V 训练时间超预期 | 用原 repo 预训练 VLM 权重做推理分析 |
| Attention weights 提取不出来 | 在 model.py 加一行 self._attn_weights = attn_weights.detach() |
| Gemini API 限流 | 减少 prompts 数量，或改用人工评分（60个输出人工可控） |
| VLM SFT 效果太差 | 这本身就是诊断结果——"VLM-SFT在小模型上效果有限"也是有价值的发现 |
| 某些可视化效果不理想 | 优先保证 Must-have 10张图，Nice-to-have 可以砍 |
| 权重文件名对不上 | 先 ls out/*.pth 确认实际文件名，再改脚本 |
