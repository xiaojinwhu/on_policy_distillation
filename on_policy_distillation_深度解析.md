# On-Policy Distillation：模型训练的第三条道路

> *如果AI训练是一场马拉松，传统方法让你要么自己摸索着跑（RL），要么只是背诵教材（SFT）。On-Policy Distillation 则是：你自己跑，但教练在每个转角实时指导——这才是冠军的训练方式。*

---

## 📖 引言：为什么这篇论文值得关注？

On-Policy Distillation 并非全新概念——早在 2015 年就有相关工作（如 DAGGER 算法）。但为什么它在 2025 年突然成为热点？

### Thinking Machines Lab 的明星阵容

2025 年 2 月，一个神秘的实验室悄然浮出水面：**Thinking Machines Lab**。它的创始团队堪称 AI 界的"复仇者联盟"：

- **Barret Zoph**：前 OpenAI 研究副总裁（后训练）
- **Lilian Weng**：前 OpenAI 副总裁
- **John Schulman**：OpenAI 联合创始人（曾短暂加入 Anthropic）
- **Bob McGrew**：前 OpenAI 首席研究官（顾问）
- **Alec Radford**：前 OpenAI 首席研究员（顾问）

这个团队从 OpenAI、Meta AI、Mistral AI 挖来约 30 名顶尖研究人员，并于 2025 年 10 月发布了 **Tinker API**——一个专门用于微调开源模型的训练平台。

### 为什么是现在？

Qwen3 技术报告显示：使用 On-Policy Distillation，以 **RL 十分之一的成本**，在 AIME'24 数学竞赛上达到了更高的 74.4% 准确率（RL 为 67.6%）。这个结果让业界意识到：

> **密集监督 + 策略采样 = 性价比之王**

这篇文章将深入解析 On-Policy Distillation 的原理、实现和应用，揭示它如何在推理、个性化、持续学习等场景中大放异彩。

---

## 🏗️ LLM 训练的三阶段范式

在深入 On-Policy Distillation 之前，我们需要理解 LLM 训练的完整流程：

```
┌─────────────────────────────────────────────────────────┐
│  Pre-training（预训练）                                   │
│  目标：语言能力、广泛推理、世界知识                          │
│  数据：大规模通用文本（数万亿 tokens）                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Mid-training（中期训练）                                 │
│  目标：领域知识（代码、医学、内部文档等）                     │
│  数据：领域特定数据（数千亿 tokens）                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Post-training（后训练）★ 本文重点                        │
│  目标：指令遵循、数学推理、对话能力                          │
│  数据：高质量标注或合成数据（数百万样本）                    │
└─────────────────────────────────────────────────────────┘
```

**关键洞察**：预训练和中期训练相对成熟，但 **后训练** 仍是效率瓶颈。传统方法要么太慢（RL），要么容易失配（SFT）。On-Policy Distillation 正是为此而生。

---

## 🎯 问题的本质：后训练的两难困境

后训练的核心任务是让模型学会**特定行为**（如指令遵循、数学推理）。目前主流方法分为两大阵营，但都存在致命缺陷。

### 方法一：Off-Policy 训练（离策略）

**典型实现**：Supervised Fine-Tuning (SFT) 或 Off-Policy Distillation

#### 工作流程

```
教师模型（如 GPT-4）
    ↓ 生成高质量示例
数据集：{"prompt": "5+(2×3)?", "response": "2×3 equals 6, add 5 to get 11."}
    ↓ 学生模型学习模仿
学生模型更新 → 逐 token 匹配教师分布
```

#### 优缺点分析

| 维度 | 描述 |
|------|------|
| ✅ **优点** | **密集反馈**：每个 token 都有监督信号，学习效率高 |
| ❌ **缺点** | **分布失配**：学生只在教师的"舒适区"学习，无法从自己的错误中恢复 |

**具体例子**：假设学生模型在推理时生成了 `5 + 2 is 7`（错误的第一步），但训练数据中教师从未犯过这个错误。学生将陷入未知状态，后续步骤会越来越偏离，导致 **复合误差 (Compounding Error)**。

![03](img/03.png)
*图：Off-Policy Distillation 示意——学生在教师轨迹上学习*

---

### 方法二：On-Policy 训练（在策略）

**典型实现**：Reinforcement Learning (RL)，如 PPO、GRPO

#### 工作流程

```
学生模型自己采样
    ↓ 生成完整推理链
"5 + 2 is 7, and 7 × 3 is 21."
    ↓ 奖励模型评分
reward = 0（答案错误）
    ↓ 策略梯度更新
学生模型调整策略，降低该轨迹概率
```

#### 优缺点分析

| 维度 | 描述 |
|------|------|
| ✅ **优点** | **相关性强**：学生在自己的分布下学习，能处理自己会遇到的错误 |
| ❌ **缺点** | **稀疏反馈**：只有最终答案的对错信号，O(1) bits per episode |

**信息论视角**：假设推理链有 N 个 token，RL 只传递 **1 bit 信息**（对/错），而每个 token 可能需要数 bits 来精确指导。效率极低。

![02](img/02.png)
*图：RL 训练示意——学生只知道 "21" 是错的，但不知道错在哪*

---

### 核心矛盾：采样 vs 反馈的两难

| 方法 | 采样来源 | 反馈密度 | 问题 |
|------|---------|---------|------|
| **SFT / Off-Policy** | 教师分布 | 密集（O(N) bits） | 学生实际会遇到的错误未覆盖 |
| **RL / On-Policy** | 学生分布 | 稀疏（O(1) bit） | 梯度噪声大，收敛慢 |

**类比理解**：
- **Off-Policy**：看棋谱学棋——你看的是大师走的每一步（密集），但从未见过自己会犯的低级错误
- **On-Policy RL**：自己下棋只看输赢——你知道输了（稀疏），但不知道是第 5 步还是第 50 步导致的
- **On-Policy Distillation**：自己下棋 + 大师实时点评——你走自己的棋（on-policy），但每一步都有详细指导（密集）

---

### 🤔 思考题

<details>
<summary><b>Q1: SFT 是 off-policy 还是 on-policy？</b></summary>

**答案**：Off-policy。SFT 训练数据是外部采样（人工标注或教师模型生成），而非学生模型自己的分布。
</details>

<details>
<summary><b>Q2: GRPO 类 RL 是 off-policy 还是 on-policy？</b></summary>

**答案**：On-policy（近似）。GRPO 每次迭代都从当前策略采样，然后立即更新策略。虽然有 importance sampling 修正，但采样分布与更新策略高度相关。
</details>

<details>
<summary><b>Q3: 过程奖励模型 (PRM) 和 On-Policy Distillation 有什么区别？</b></summary>

**核心差异**：
- **PRM**：需要人工标注每一步的对错（成本高），奖励仍是离散的（0/1）
- **On-Policy Distillation**：利用教师模型的 **概率分布** 作为连续奖励，无需人工标注
- **相似点**：都提供逐步监督（dense reward）
</details>

---

### 突破口：能否两全其美？

我们的目标很明确：

> **在学生自己的轨迹上（on-policy），提供逐 token 的密集指导（dense reward）**

这就是 **On-Policy Distillation** 的核心思想。接下来，我们将看到它如何优雅地解决这一矛盾。

---

## 💡 解决方案：On-Policy Distillation——两全其美的训练范式

### 方法对比：打破不可能三角

| 方法 | 采样来源 | 反馈密度 | 适用场景 |
|------|---------|---------|---------|
| **Supervised Fine-Tuning** | Off-policy | Dense (O(N) bits) | 有大量高质量标注数据 |
| **Reinforcement Learning** | On-policy | Sparse (O(1) bit) | 有明确奖励信号（如代码执行结果） |
| **On-Policy Distillation** | On-policy | Dense (O(N) bits) | 有强教师模型，需高效训练 |

**核心创新**：On-Policy Distillation 巧妙地结合了两种方法的优点：

```
学生模型采样自己的轨迹（on-policy）
    ↓
"5 + 2 is 7, and 7 × 3 is 21."
    ↓
教师模型对每个 token 评分（dense）
    ↓
[40%, 80%, 5%, 15%, 99%, 99%, 40%, 80%, 90%, 99%, 70%, 99%, 99%]
         ^^^  ^^^  ← 这些步骤错了！
    ↓
学生模型精确知道在哪里犯错，如何改进
```

![04](img/04.png)
*图：On-Policy Distillation 示意——在学生轨迹上提供逐 token 的教师指导*

---

### 🎮 类比：如何成为象棋高手？

想象你要学习下象棋，有三种训练方式：

#### 方法 A：看棋谱（Off-Policy）
- 📚 你观看卡斯帕罗夫的经典对局，学习每一步的走法
- ✅ **优点**：每一步都有"标准答案"
- ❌ **缺点**：但你是新手，实际对局时会走出大师从未走过的烂棋。这时候你就懵了，因为棋谱里从来没有从"烂棋"恢复的例子

#### 方法 B：自己下棋（On-Policy RL）
- 🎲 你自己下 100 局，只看最终输赢
- ✅ **优点**：你确实在自己的水平上练习
- ❌ **缺点**：你输了，但不知道是第 5 步、第 15 步还是第 50 步导致的。效率太低

#### 方法 C：实时教练（On-Policy Distillation）
- 👨‍🏫 **你自己下棋，但教练在旁边对每一步打分**：
  - "这一步是失误（5%）"
  - "这一步还不错（80%）"
  - "这一步精妙！（99%）"
- ✅✅ **两全其美**：在你的水平上练习 + 每一步都有指导

**Chess.com 的实现**：正是这种模式！引擎会对你的每一步棋标注为"失误"、"不准确"或"精彩"。

![棋盘分析示例](img/01.png)

---

### 💰 为什么不直接用教师模型？

**Q: 既然教师这么强，为什么还要训练学生？**

**A: 小模型经过专项训练，在特定领域可以超越通用大模型，且有独特优势：**

| 维度 | 小模型 | 大模型 |
|------|--------|--------|
| **推理成本** | 低（可部署在边缘设备） | 高（需要大规模集群） |
| **隐私安全** | 可本地部署，数据不出域 | 通常需要API调用 |
| **持续学习** | 易于更新和微调 | 重训练成本极高 |
| **专项性能** | 在训练领域可超越大模型 | 泛化能力强但不专精 |

**经验法则**：Qwen3-8B 经过专项训练后，在数学推理上可以超越未经训练的 Qwen3-32B。

---

### 🔬 实现细节：如何将理论变为代码

#### 损失函数：Reverse KL Divergence

On-Policy Distillation 的核心是 **逐 token 的反向 KL 散度**：

```
KL(π_θ || π_teacher) = E_{x~π_θ} [log π_θ(x_{t+1}|x_{1..t}) - log π_teacher(x_{t+1}|x_{1..t})]
```

**直观理解**：
- `π_θ(x)`：学生认为下一个 token 是 x 的概率
- `π_teacher(x)`：教师认为下一个 token 是 x 的概率
- **Reverse KL = 0**：学生完全模仿教师
- **Reverse KL 越大**：学生越偏离教师

![05](img/05.png)

**为什么用 Reverse KL 而非 Forward KL？**

| 特性 | Reverse KL | Forward KL |
|------|-----------|-----------|
| **模式覆盖** | Mode-seeking（专注单一策略） | Mode-covering（分散在多个策略） |
| **适用任务** | 精确推理（数学、代码） | 创意生成（对话、写作） |
| **与 RL 的兼容性** | 天然对齐（RL 也是 Reverse KL） | 需要额外调整 |

**Mode-seeking 特性**：Reverse KL 让学生学习教师的**一种**高质量策略，而非尝试覆盖所有可能的策略。这对数学推理等需要确定性的任务至关重要。

---

#### 伪代码：四步实现 On-Policy Distillation

```python
# 1. 初始化教师模型客户端
teacher_client = create_sampling_client(
    base_model="Qwen3-32B",
    model_path="path/to/teacher"
)

# 2. 学生模型采样轨迹（on-policy）
trajectories = student_model.sample(prompts)  # 形状: [batch, seq_len]
student_logprobs = trajectories.logprobs      # 学生自己的 log 概率

# 3. 教师模型对学生轨迹评分（dense reward）
teacher_logprobs = teacher_client.compute_logprobs(trajectories)
reverse_kl = student_logprobs - teacher_logprobs  # 逐 token 的 KL

# 4. 策略梯度更新（类似 RL 的 advantage）
advantages = -reverse_kl  # KL 越大，惩罚越大
loss = importance_sampling_loss(trajectories, advantages)
loss.backward()
```

**关键特性**：
- **无需等待完整序列**：可以用部分轨迹训练（节省内存）
- **无需独立奖励模型**：教师的 logprobs 即是奖励
- **高度可并行**：教师前向传播与学生训练可并行

![07](img/07.png)

---

#### 💡 一个真实例子：SimpleBench 错误分析

**问题**：将 4 个冰块放入煎锅，第一分钟结束时有多少个冰块？

**学生答案（Qwen3-4B-Instruct）**：
> "第一分钟结束时，煎锅里还有 **20 个冰块**。"

**教师评分（Qwen3-235B）**：

![06](img/06.png)

**分析**：
- 深红色 tokens（"total", "in the", "**20**"）：这些词导致了错误推理
- 浅色 tokens：相对合理的步骤
- **关键发现**：教师精确定位了"分叉点"（forking tokens）——正是 "total" 这个词让推理走向了纯数学计算，而忽略了物理常识（冰会融化）

**正确答案应该是 0**，因为冰块会在煎锅中融化。

---

### ⚡ 计算效率：为什么比 RL 快 10-100 倍？

| 效率来源 | 具体表现 |
|---------|---------|
| **无需完整 rollout** | 可用部分轨迹训练，节省采样成本 |
| **教师只需前向传播** | 不需要反向传播，可用 FP16/INT8 加速 |
| **学生成本低** | 轨迹由小模型生成，而非大模型 |
| **密集信号降低方差** | 每个 token 都有监督，梯度更稳定，可用更小 batch size |

**具体数字**（来自论文实验）：
- **GPU 时间**：On-Policy Distillation 1,800 小时 vs RL 17,920 小时（**10倍**加速）
- **FLOPs**：9-30 倍计算效率提升（取决于是否计入数据集生成成本）

---

### 🔮 未来方向：混合奖励

论文提到，可以将蒸馏的逐 token 奖励与环境奖励（如代码执行结果）结合：

```python
reward_total = α * reward_distillation + β * reward_environment
```

这在代码生成、工具使用等场景中尤其有前景。

---

## 📊 实验验证：理论如何落地？

论文在两个核心场景验证了 On-Policy Distillation 的有效性：
1. **数学推理**：在 AIME'24 竞赛题上训练 Qwen3-8B
2. **个性化助手**：让模型同时掌握内部知识和指令遵循能力

---

### 实验一：数学推理训练（Distillation for Reasoning）

**实验设置**：
- **学生模型**：Qwen3-8B-Base
- **教师模型**：Qwen3-32B
- **数据集**：OpenThoughts-3（由 QwQ-32B 生成的推理链数据）
- **评估基准**：AIME'24（美国数学邀请赛，高中难度）

#### 阶段 1：Off-Policy Distillation（预热）

首先在 40 万条教师生成的数据上进行 SFT，建立基础能力：

![08](img/08.png)
*图：Off-Policy Distillation 的性能曲线（Log-Linear Scaling）*

**关键发现**：
- 初期进步很快（前 10 万样本），之后进入对数增长
- 要从 60% 提升到 70%，需要外推到 **200 万样本**（5 倍数据量）
- LoRA 虽然降低遗忘，但在大规模训练中落后全量微调

#### 阶段 2：三种后训练方法对比

从 SFT-400K checkpoint 出发，尝试三种方式提升性能：

| 方法 | AIME'24 | GPQA-Diamond | GPU 时间 | 计算效率 |
|------|---------|--------------|----------|---------|
| **Off-policy SFT（外推）** | ~70% | - | ~18,000h | 1× 基准 |
| **Reinforcement Learning** | 67.6% | 61.3% | 17,920h | ≈1× |
| **On-Policy Distillation** | 74.4% | 63.3% | 1,800h | **10×** |

*数据来源：Qwen3 Technical Report, Table 21*

**震撼结论**：
- On-Policy Distillation 不仅 **更快（10倍）**，而且 **更强（+6.8% 绝对提升）**
- 仅需 **150 步**训练即达到 70%（vs SFT 需要 5 倍数据）

#### 阶段 3：详细 FLOPs 分析

从计算量角度进一步分解：

| 方法 | AIME'24 | 教师 FLOPs | 学生 FLOPs | 相比 SFT-2M |
|------|---------|-----------|-----------|------------|
| **SFT-400K（起点）** | 60% | 8.5×10²⁰ | 3.8×10²⁰ | - |
| **SFT-2M（外推）** | ~70% | 3.4×10²¹ | 1.5×10²¹ | 1× |
| **RL** | 68% | - | - | ≈1× |
| **On-Policy Distill** | 70% | 8.4×10¹⁹ | 8.2×10¹⁹ | **9-30×** |

**效率来源分解**：
- **基础效率**：9× （假设 SFT 数据集已生成）
- **完全效率**：30× （计入教师模型生成数据的成本）
- **GPU 时间效率**：18× （因为教师 logprob 计算可高度并行）

![09](img/09.png)
*图：On-Policy Distillation 的训练曲线——150 步内达到 70%*

**有趣发现**：
- LoRA 在 SFT 中落后 13%，但在 On-Policy Distillation 中只落后 6%
- 原因：密集反馈降低了参数更新的方差，LoRA 的低秩约束影响更小

---
### 实验二：个性化助手训练（Distillation for Personalization）

**场景描述**：训练一个既懂公司内部知识，又会遵循指令的企业助手。

#### 核心挑战：知识 vs 行为的权衡

企业助手需要同时满足两个目标：

| 维度 | 要求 | 评估方法 |
|------|------|---------|
| **领域知识** | 回答公司内部文档相关问题 | Internal QA Eval（自建） |
| **助手行为** | 遵循复杂指令（如格式、语气） | IF-Eval（标准基准） |

**困境**：在公司文档上进行 Mid-Training 会导致 **灾难性遗忘 (Catastrophic Forgetting)**——模型学会了知识，却忘记了如何遵循指令。

---

#### 实验设计：三阶段训练流程

```
阶段 1: 选择基础模型
Qwen3-8B-Instruct（已具备指令遵循能力）
↓
阶段 2: Mid-Training（学习内部知识）
在公司文档 + Tulu3 对话数据混合训练
↓
阶段 3: On-Policy Distillation（恢复行为）
用原始 Qwen3-8B 作为教师，恢复指令遵循能力
```

---

#### 阶段 1：灾难性遗忘的量化

在不同比例的数据混合下训练，观察知识和行为的权衡：

![10](img/10.png)
*图：不同数据混合比例的效果——横轴为文档占比*

**关键发现**：
- **100% 文档**：知识从 18% → 43%（✅），但指令遵循从 85% → 45%（❌ 崩溃）
- **70% 文档 + 30% 对话**：知识 36%（略降），指令遵循 79%（仍有损失）
- **无完美配比**：任何混合都无法同时保持两项能力

**为什么混合数据不够？**
- SFT 训练会改变模型的参数分布，即使混入对话数据，也无法阻止 RL 训练过的"脆弱子网络"被覆盖
- IF-Eval 性能在训练中持续下降，最终无法恢复

![11](img/11.png)
*图：IF-Eval 在 Mid-Training 过程中持续下降*

---

#### 阶段 2：LoRA 能否缓解遗忘？

**实验**：用 LoRA（rank=32/128）代替全量微调，限制参数更新范围。

**结果**：
- ✅ LoRA 确实减少遗忘（IF-Eval 从 45% → 65%）
- ❌ 但也减少学习（Internal QA 从 43% → 35%）

**结论**：LoRA 是权衡，而非解决方案。

---

#### 阶段 3：On-Policy Distillation 的奇迹

**核心思路**：用训练前的 Qwen3-8B-Instruct 作为教师，在 Tulu3 对话数据上进行 On-Policy Distillation，"重新唤醒"丢失的指令遵循能力。

**训练流程**：

```python
# 教师：训练前的原始模型
teacher = Qwen3-8B-Instruct (original)

# 学生：Mid-Training 后的模型（已有内部知识）
student = Qwen3-8B-Instruct (after mid-training)

# 在对话数据上 On-Policy Distillation
for prompt in Tulu3:
    trajectory = student.sample(prompt)  # 学生自己生成
    teacher_logprobs = teacher.compute_logprobs(trajectory)
    loss = kl_divergence(student, teacher_logprobs)
    loss.backward()
```

**最终结果**：

| 模型 | Internal QA（知识） | IF-Eval（行为） |
|------|--------------------:|----------------:|
| Qwen3-8B（基线） | 18% | **85%** |
| + Mid-Training（70%文档） | 36% | 79% ↓ |
| + **On-Policy Distillation** | **41%** ↑ | **83%** ↑ |

**震撼效果**：
- ✅ **知识不降反升**：36% → 41%（+5%）
- ✅ **行为几乎完全恢复**：79% → 83%（接近原始 85%）
- **机制解释**：对话能力的恢复带来了"正向迁移"，提升了模型的整体推理能力

---

#### 洞察：持续学习的新范式

On-Policy Distillation 开辟了一条 **增量学习 (Continual Learning)** 的新路径：

```
循环迭代：

[新知识学习] → Mid-Training on new data
    ↓（可能遗忘行为）
[行为恢复] → On-Policy Distillation with old checkpoint
    ↓（保留知识，恢复行为）
[评估] → 检查知识 + 行为双指标
    ↓
回到 [新知识学习]
```

**关键优势**：
- 无需重新训练整个模型
- 可以用任意版本的历史 checkpoint 作为教师
- 适用于生产环境的模型持续更新

**相关工作**：Cobbe et al. (2021) 的相变学习 (Phase-Alternating Learning)。

---

## 🧠 深度讨论：为什么 On-Policy Distillation 如此有效？

### 讨论一：密集监督 = 50-100× 计算效率提升

#### 信息论视角

**RL vs Distillation 的信息传递效率**：

| 方法 | 每 Episode 信息量 | 有效学习速度 |
|------|------------------|------------|
| **Reinforcement Learning** | O(1) bit（对/错） | 基准 |
| **On-Policy Distillation** | O(N) bits（N 个 token 的概率分布） | **N 倍** |

**极端实验**：从 Qwen3-8B-Base 开始，LoRA rank=128：
- **RL 训练**：70 步达到 AIME'24 = 29%
- **Distillation 训练**：10 步达到同样性能（**7× 加速**）
- **累计效率**：考虑 batch size、context length 等因素，**50-100×** 计算节省

**为什么 Dense Reward 这么强？**
- RL 的梯度方差正比于 1/√(信息量)
- Distillation 提供 N 倍信息 → 方差降低 √N 倍 → 可用更小 batch size → 内存节省

**历史佐证**：Lightman et al. (2023) 的过程监督实验也显示，逐步奖励比最终奖励效率高一个数量级。

---

### 讨论二：数据复用效率——从"记忆答案"到"学习分布"

#### 多轮采样实验

**极端测试**：仅用 **1 个 prompt**，能训练出模型吗？

**实验设置**：
- 从 Qwen3-8B-Base 开始
- 随机选 1 个 AIME 题目
- 每步采样 256 条学生轨迹，训练 20 步（共 5,120 条重复样本）

**结果**：
- RL：无法收敛（严重过拟合，记住了这道题的答案）
- **On-Policy Distillation**：达到教师模型性能（AIME'24 ≈ 29%）

**原理**：
- **RL 目标**：最大化 P(正确答案)，容易记住特定 token 序列
- **Distillation 目标**：最小化 KL(学生||教师)，学习的是概率分布，而非单一答案

**数学解释**：
```
RL:     max  E_{x~π_θ}[R(x)]              → 找最高奖励的单一路径
Distill: min  E_{x~π_θ}[KL(π_θ||π_teacher)] → 匹配整个分布
```

**实践意义**：
- 可以在同一 prompt 上重复训练多轮（数据效率提升）
- 适合小数据集场景（如企业内部文档）

---

### 讨论三：RL 在"语义策略空间"搜索 vs Pre-Training 在"参数空间"搜索

#### 两种搜索范式的对比

| 维度 | Pre-Training | RL + Distillation |
|------|-------------|------------------|
| **搜索空间** | 高维参数空间（10⁹-10¹² 维） | 语义策略空间（离散的高层策略） |
| **搜索机制** | 梯度下降（确定性） | 采样 + 信用分配（随机性） |
| **计算瓶颈** | 梯度计算（expensive） | 采样 + 奖励计算（cheap） |
| **知识表示** | 网络权重（难以迁移） | 语义行为（易于蒸馏） |

#### RL "偶然发现"好策略，Distillation "快速复制"

**类比**：
- **RL**：像进化算法，随机变异 + 自然选择，可能花 10,000 代才找到某个突变
- **Distillation**：像基因工程，直接将"优良基因"（教师策略）注入学生

**为什么 Distillation 不需要建模 RL 的"探索过程"？**

论文的科研类比：
> 科学研究需要大量试错才能发现新定理，但教授可以用一节课就教会学生这个定理。探索和传播是两个不同的过程。

**实践启示**：
- RL 适合"探索"新策略（如 AlphaGo 的自对弈）
- Distillation 适合"传播"已知策略（如训练小模型复现大模型能力）

---

### 讨论四：SFT 的隐藏陷阱——"On-Policy" 数据也会导致 Off-Policy 训练

#### 反直觉实验：自蒸馏会退化性能！

**实验**：让 Qwen3-32B 在自己采样的数据上 SFT（完全 on-policy！）

**预期**：性能应该保持不变（KL=0 in expectation）

**实际结果**：
- IF-Eval 从 85% 暴跌到 76%
- 训练越久，性能越差

**原因分析**：
1. **有限批次的方差**：虽然期望上 KL=0，但每个 mini-batch 都有随机性
2. **误差累积**：微小的梯度更新 → 策略漂移 → 下一批数据变成"轻度 off-policy" → 更大漂移...
3. **长序列的放大效应**：在 IF-Eval（需要长上下文的指令遵循）上，误差沿序列累积

**On-Policy Distillation 的优势**：
- 教师固定不变，学生始终"校准"到教师
- 即使有方差，最终收敛到教师分布（而非漂移）

**数学直觉**：
```
SFT on own samples:  π_{t+1} ← π_t (no gradient in expectation, but variance)
On-Policy Distill:   π_{t+1} ← π_teacher (always pulled back)
```

---

### 讨论五：持续学习的新工具

On-Policy Distillation 最令人兴奋的应用可能是 **终身学习 (Lifelong Learning)**：

#### 传统方法的困境

| 方法 | 问题 |
|------|------|
| **Fine-Tuning** | 灾难性遗忘（如实验二所示） |
| **LoRA** | 能力受限，难以学习新知识 |
| **Replay Buffer** | 需存储大量历史数据，成本高 |
| **Elastic Weight Consolidation** | 依赖 Fisher 信息矩阵，计算昂贵 |

#### On-Policy Distillation 的新范式

```python
# 伪代码：持续学习循环
model = load_pretrained_model()
history_checkpoints = []

while True:
    # 阶段 1: 学习新任务
    new_task_data = get_new_task_data()
    model = mid_train(model, new_task_data)

    # 阶段 2: 恢复旧能力
    old_task_data = sample_old_tasks()
    for old_ckpt in history_checkpoints:
        model = on_policy_distill(
            student=model,
            teacher=old_ckpt,
            data=old_task_data
        )

    # 阶段 3: 评估 + 保存
    if eval(model) > threshold:
        history_checkpoints.append(model.clone())
```

**关键洞察**：
- 历史 checkpoint 本身就是"记忆"，无需额外存储数据
- On-Policy 采样确保在当前策略下恢复能力
- 可以选择性地"唤醒"特定版本的能力

**相关工作**：
- Cobbe et al. (2021): Phase-Alternating Learning
- Chen et al. (2024): On-Policy SFT for Continual Learning

---

## 💻 实践指南：如何使用 Tinker 复现论文结果

### 官方资源

- **Tinker Cookbook**：https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
- **代码示例**：完整的 On-Policy Distillation 实现

![14](img/14.png)

### 训练流程总结

#### 场景一：数学推理模型

```
Step 1: Off-Policy Distillation（预热）
  - 数据：OpenThoughts-3（40万样本）
  - 学生：Qwen3-8B-Base
  - 目标：AIME'24 ≈ 60%

Step 2: On-Policy Distillation（冲刺）
  - 教师：Qwen3-32B
  - 数据：学生自己采样 + 教师评分
  - 训练：150 steps
  - 目标：AIME'24 ≈ 70%
```

#### 场景二：企业助手

```
Step 1: Mid-Training（知识注入）
  - 数据：70% 内部文档 + 30% Tulu3 对话
  - 目标：Internal QA ≈ 36%，IF-Eval ≈ 79%

Step 2: On-Policy Distillation（行为恢复）
  - 教师：训练前的 Qwen3-8B-Instruct
  - 数据：Tulu3 对话（学生采样）
  - 目标：Internal QA ≈ 41%，IF-Eval ≈ 83%
```

---

## 🎯 核心要点总结

### 方法本质

| 维度 | 描述 |
|------|------|
| **核心思想** | 在学生自己的轨迹上，提供教师的逐 token 指导 |
| **采样** | On-policy（学生模型采样） |
| **反馈** | Dense（每个 token 都有 KL 奖励） |
| **损失函数** | Reverse KL（mode-seeking） |

### 性能对比

| 指标 | vs SFT | vs RL |
|------|--------|-------|
| **计算效率** | 9-30× | 10-100× |
| **数据效率** | 可重复采样同一 prompt | 信息密度高 N 倍 |
| **性能** | 避免分布失配 | 更高准确率 |

### 适用场景

✅ **适合**：
- 有强大教师模型（如 GPT-4、Claude、Qwen-32B）
- 需要精确推理（数学、代码、逻辑）
- 持续学习、个性化微调

❌ **不适合**：
- 教师模型本身不准确
- 创意生成任务（可能需要 diversity）
- 学生初始能力太弱（需先 off-policy 预热）

### 局限与未来方向

| 局限 | 潜在解决方案 |
|------|------------|
| 依赖高质量教师 | 探索 ensemble teachers |
| 冷启动问题 | 结合 off-policy distillation |
| 单一教师限制 | 混合多个教师（如 Qwen-32B + GPT-4） |
| 未整合环境奖励 | **混合奖励**：α·KL + β·reward_env |

---

## 🔮 未来展望

On-Policy Distillation 开启了后训练的新时代，但这只是开始：

1. **混合奖励机制**：将蒸馏与环境反馈（如代码执行、工具调用）结合
2. **多教师集成**：从多个专家模型蒸馏（类似 Mixture of Experts）
3. **自适应教师选择**：根据学生能力动态选择教师难度
4. **在线持续学习**：生产环境中实时更新模型
5. **跨模态扩展**：将 On-Policy Distillation 应用于视觉-语言模型

**最激动人心的可能性**：随着 On-Policy Distillation 的普及，我们或许能让每个人都拥有一个"个性化AI助手"——它既具备 frontier 模型的能力，又能本地部署、持续学习、尊重隐私。这才是 AI 民主化的真正未来。

---

## 📚 参考文献

本文基于以下工作：

1. **On-Policy Distillation** - Kevin Lu et al., Thinking Machines Lab, 2025
2. **Qwen3 Technical Report** - Qwen Team, 2025
3. **DAGGER** - Ross et al., 2011
4. **LoRA Without Regret** - Prior work on efficient fine-tuning
5. **Process Reward Models** - Lightman et al., 2023

---

*感谢阅读！如果你对 On-Policy Distillation 有任何疑问或想法，欢迎交流讨论。*