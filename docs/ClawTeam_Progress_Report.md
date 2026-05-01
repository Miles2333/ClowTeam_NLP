# ClawTeam 项目进展汇报

> **项目名称**：ClawTeam — 基于 Harness Engineering 与领域微调的医疗多智能体协作诊疗系统
> **课题来源**：基于开源框架 miniOpenClaw 的医疗领域改造与扩展
> **汇报阶段**：MVP 完成 + 部分训练任务完成

---

## 一、项目背景与动机

### 1.1 现有方案的局限

| 现有方案 | 局限 |
|---------|------|
| **GPT-4 / Claude 直接咨询** | 通用对话能力强，但**缺乏专科分工**；安全性依赖闭源策略；无法本地部署，隐私风险 |
| **HuatuoGPT / DISC-MedLLM** | 单 Agent 架构，无角色分工，**无主动安全守卫** |
| **Harness 工程方案（Claude Code 类）** | 单 Agent + 工具调用，**未做领域微调** |

### 1.2 ClawTeam 的核心定位

> **将 Harness Engineering（工具/记忆/Skill）与领域微调（LoRA + 多模态）结合，构建一个专科分工、共享记忆、安全可控的医疗多智能体系统。**

---

## 二、系统架构

### 2.1 整体架构图

```
                     ┌──────────────┐
                     │   用户输入    │
                     └──────┬───────┘
                            ▼
                  ┌─────────────────────┐
                  │  Guardian 安全守卫   │  ←── 训练后的 BERT 5 类分类器
                  │   (5 类风险识别)    │       (safe/injection/privilege/
                  └─────────┬───────────┘        privacy/dangerous)
                  危险      │      安全
                   ┌────────┘      └────────┐
                   ▼                        ▼
              [拦截+提示]          ┌──────────────────┐
                                   │  Coordinator     │
                                   │  (协调器/路由器)  │ ←── 训练后的 BERT 多标签分类
                                   └──────┬───────────┘
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    ┌──────────┐    ┌──────────┐    ┌──────────┐
                    │ 主治医生 │    │  药师    │    │  影像科  │
                    │ Physician│    │Pharmacist│    │Radiolog. │
                    │ (LoRA)   │    │ (LoRA)   │    │(VL LoRA) │
                    └─────┬────┘    └─────┬────┘    └─────┬────┘
                          │               │               │
                          └───────┬───────┴───────┬───────┘
                                  ▼               ▼
                         ┌────────────────────────────────┐
                         │   共享长期记忆                  │
                         │ (PostgreSQL + pgvector + BM25) │
                         └────────────┬───────────────────┘
                                      ▼
                            ┌─────────────────┐
                            │  意见融合协调器  │
                            │  (LLM 综合输出) │
                            └────────┬────────┘
                                     ▼
                          ┌─────────────────────┐
                          │ 综合诊疗建议        │
                          │ (各角色意见 + 综合)  │
                          └─────────────────────┘
```

### 2.2 组件职责说明

| 组件 | 角色 | 实现方式 |
|------|------|---------|
| **Guardian** | 输入安全过滤 | BERT 5 类分类（训练得到） |
| **Coordinator** | 路由决策 + 意见融合 | BERT 多标签分类 + LLM 综合 |
| **主治医生 Agent** | 诊断路径、鉴别诊断 | Qwen2.5 + LoRA 微调 |
| **临床药师 Agent** | 用药安全、相互作用 | Qwen2.5 + LoRA 微调 |
| **影像科 Agent** | 影像检查建议、解读 | Qwen2-VL + 多模态 LoRA |
| **共享记忆** | 跨会话上下文 | pgvector + BM25 混合检索 |

---

## 三、四个创新点

### 创新点 1：多智能体协作架构 + 共享长期记忆
- **现状**：医疗 LLM 大多是单 Agent，记忆只在当前会话内
- **本项目**：3 个专科 Agent 并行会诊，**共享同一记忆池**（基于 miniOpenClaw memory_v2）
- **价值**：跨会话患者级跟踪，避免重复问诊

### 创新点 2：医疗领域专属安全守卫
- **现状**：GPT-4 的 safety 依赖闭源策略；Llama Guard 只覆盖通用攻击
- **本项目**：**5 类医疗特异性风险识别**：
  - `injection` 提示注入
  - `privilege` 越权诊断（要求开处方）
  - `privacy` 隐私诱导（询问其他患者）
  - `dangerous` 危险医疗（自残/超剂量）
  - `safe` 正常咨询
- **价值**：通过监督学习训练专属 BERT 分类器，可解释、可定制

### 创新点 3：Harness × Training 混合架构
- **现状**：harness engineering 路线（Claude Code）不微调；微调路线（Med-PaLM）不用工具
- **本项目**：**两者结合**——
  - LoRA 微调让小模型学到专科风格
  - 保留工具调用能力（read_file、search_memory 等）
  - 保留 Skill 系统（如临床指南可热更新）
- **价值**：3B 模型在特定任务上性能逼近 GPT-4，同时保留可扩展性

### 创新点 4：多模态医学视觉-语言模型微调
- **现状**：医疗多模态多用 GPT-4V API；少有公开的领域微调
- **本项目**：用 Qwen2-VL + 医学影像数据集（SLAKE / VQA-RAD）做 LoRA 微调
- **价值**：影像科 Agent 能**真正读图**，不是文字描述猜测

---

## 四、已完成的工作

### 4.1 系统架构（MVP 已上线）

```
✅ 后端 FastAPI + LangGraph 多 Agent 框架
✅ 协调器 + 3 角色 Agent（physician / pharmacist / radiologist）
✅ 角色 prompt 工程（workspace/roles/*.md）
✅ Guardian 中间件（医疗增强版）
✅ 4 组实验模式切换：
   - single（单 Agent 基线）
   - multi_no_memory（多角色无记忆）
   - multi_memory（多角色 + 记忆）
   - multi_full（多角色 + 记忆 + Guardian）
✅ 实验日志框架（JSONL 落盘）
✅ 前端多角色意见展示 + 推荐气泡
✅ 共享记忆模块（v2: pgvector + BM25 + RRF 融合）
✅ GitHub 仓库：https://github.com/Miles2333/ClowTeam_NLP
```

### 4.2 训练任务进展

| 任务 | 模型 | 数据 | 状态 | 关键指标 |
|------|------|------|------|---------|
| **01 路由分类器** | bert-base-chinese (110M) | 1002 条（LLM 生成） | ✅ 完成 | macro_F1 = **1.000** |
| **02 Guardian 守卫** | bert-base-chinese (110M) | 1500 条（5 类均衡） | ✅ 完成 | macro_F1 = **0.986** |
| **03 主治医生 LoRA** | Qwen2.5-0.5B | 800 条问答对 | ✅ 完成 | Loss = **0.79** |
| **04 药师 LoRA** | Qwen2.5-0.5B | 800 条问答对 | 🚧 数据生成中 | - |
| **05 影像科 VL LoRA** | Qwen2-VL-2B + SLAKE | 2000 条 VQA | ⏳ 待开始 | - |

### 4.3 关键实验结果

#### 结果 1：路由分类器
- 1000+ 条标注数据训练，5 epoch，**测试集准确率 100%**
- ⚠️ 100% 是因为合成数据中关键词与标签强关联，需要后续真实数据验证

#### 结果 2：Guardian 安全守卫（混淆矩阵）
```
            safe inject privil privac danger
safe         62    0     0     0     0    ← recall 100%（无误拦）
injection     0   21     0     0     0    ← 完美
privilege     0    0    18     0     0    ← 完美
privacy       0    0     0    20     0    ← 完美
dangerous     1    0     1     0    27    ← 1 条漏拦（重点关注）
```

**关键发现**：dangerous 类有 1 条漏拦截，是医疗系统**最危险的错误类型**，将作为论文中"非对称代价学习"的研究方向。

#### 结果 3：主治医生 LoRA（推理样例）
学到了规范的【主诉分析】→【鉴别诊断】→【建议检查】→【初步治疗】结构化输出，但医学准确性受限于 0.5B 基座模型规模。

> **重要观察**：LoRA 学到了**风格和结构**，但**医学知识精度** 受限于基座模型。这本身是一个有价值的实验发现（"风格 vs 知识"的解耦特性）。

---

## 五、进度时间线

```
         ┌──── 已完成 ──┐    ┌── 进行中 ──┐    ┌─── 计划中 ───┐
2026-04 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2026-06
         │              │    │            │    │              │
         ▼              ▼    ▼            ▼    ▼              ▼
       MVP             训练  04药师LoRA  云端  论文写作      答辩
     架构搭建      01/02/03  05影像VL    训练  实验分析
                                         3B/7B
```

### 详细里程碑

| 阶段 | 时间 | 内容 | 状态 |
|------|------|------|------|
| **Phase 1: MVP** | 4月初-4月中 | 多 Agent 架构 + 前后端 | ✅ 完成 |
| **Phase 2: 数据 + 小模型训练** | 4月中-4月底 | 5 个训练任务（本地） | 🚧 60% |
| **Phase 3: 云端大模型训练** | 5月初-5月中 | Qwen2.5-3B/7B + Qwen2.5-VL-3B/7B | ⏳ 计划 |
| **Phase 4: 评测 + 消融** | 5月中-5月底 | 4 组对比 + 挑战测试集 | ⏳ 计划 |
| **Phase 5: 论文写作** | 6月初-6月中 | 结构化报告 + 投稿 | ⏳ 计划 |

---

## 六、计划中的实验（论文核心）

### 6.1 主实验：4 组对比

| 组别 | 配置 | 验证目的 |
|------|------|---------|
| A | 单 Agent | 基线 |
| B | 多 Agent 无记忆 | 验证协作架构价值 |
| C | 多 Agent + 共享记忆 | 验证共享记忆价值 |
| D | 多 Agent + 共享记忆 + Guardian | 完整方案 |

### 6.2 消融实验

| 维度 | 对比 |
|------|------|
| 路由方式 | 关键词规则 vs BERT 训练分类器 vs LLM 路由 |
| Guardian | 无 vs prompt-based vs BERT 训练分类器 |
| 角色实现 | prompt-only vs LoRA 微调 |
| 多模态 | 文字描述 vs Qwen-VL LoRA 真实图像 |

### 6.3 评测指标

- **医学正确性**：MedQA/CMExam 准确率，专家盲评
- **安全性**：攻击拦截率、误拦率、**dangerous 漏拦率（重点）**
- **协作质量**：角色意见一致性、冲突解决成功率
- **效率**：平均时延、token 消耗、工具调用次数

---

## 七、技术栈

| 层级 | 技术 |
|------|------|
| **LLM 框架** | LangChain 1.x + LangGraph（多 Agent 编排） |
| **后端** | FastAPI + SSE 流式响应 |
| **前端** | Next.js 14 + React 18 + Tailwind |
| **训练** | PyTorch 2.7 + Transformers 5.x + PEFT (LoRA) + TRL (SFT) |
| **基座模型** | 本地: Qwen2.5-0.5B + Qwen2-VL-2B；云端: Qwen2.5-3B/7B + Qwen2.5-VL-3B/7B |
| **数据库** | PostgreSQL + pgvector（向量检索）+ BM25（关键词） |
| **评估** | scikit-learn + sacrebleu + rouge-chinese |
| **追踪** | Langfuse（LLM trace）+ wandb（训练监控） |
| **部署** | 本地训练 + AutoDL 云端训练（5070 Ti / 4090） |

---

## 八、当前面临的问题与解决方案

| 问题 | 现状 | 解决方案 |
|------|------|---------|
| 100% 准确率虚高 | 路由分类器 100% 来自合成数据同分布 | 构建挑战测试集（手工真实风格） |
| 0.5B 模型医学幻觉 | 推荐"咳嗽做冠脉造影" | 云端换 3B/7B 验证；论文里讨论"风格-知识"解耦 |
| dangerous 漏拦 | Guardian 1/29 漏拦截 | 加入非对称代价；论文重点讨论 |
| 真实医学数据稀缺 | 当前依赖 LLM 生成 | 引入 CMExam、Huatuo、SLAKE 等公开数据集 |
| 显存限制 | 5070 Ti 12GB 跑 7B 紧张 | 后期切换 AutoDL 4090 24GB |

---

## 九、需要 TA 帮助 / 讨论的问题

1. **数据集补充**：是否有访问 MIMIC-CXR / PhysioNet 的资源？（需要审核账号）
2. **专家评估**：能否联系医学院学生帮忙做盲评？
3. **论文方向**：投会议（ACL/EMNLP workshop）还是期刊？建议哪个方向更合适？
4. **基座模型**：Qwen3-VL-4B 是否值得尝试（最新但教程少）？
5. **答辩节奏**：6 月底前能完成全套实验吗？

---

## 十、本次汇报总结

**已经做到的**：
- ✅ 完整 MVP 架构（多 Agent + 共享记忆 + Guardian）
- ✅ 跑通完整训练 Pipeline（数据生成 → 训练 → 评测 → 集成）
- ✅ 5 个训练任务中的 3 个已完成（路由 + 守卫 + 主治）

**正在做**：
- 🚧 药师 LoRA + 影像科多模态 LoRA
- 🚧 Guardian 漏拦案例分析

**接下来 1 个月**：
- ⏳ 云端 3B/7B 大模型训练
- ⏳ 4 组对比实验 + 消融
- ⏳ 挑战测试集构建
- ⏳ 论文初稿

---

## 附录 A：项目仓库

- **代码**：https://github.com/Miles2333/ClowTeam_NLP
- **基础架构**：基于 [miniOpenClaw](https://github.com/Miles2333/miniOpenClaw) 改造
- **当前 tag**：`v0.1-mvp`

## 附录 B：核心新增模块清单

```
backend/
├── graph/
│   ├── coordinator.py        # 协调器（路由 + 融合）
│   └── roles/                # 3 个角色 Agent
│       ├── base_role.py
│       ├── physician.py
│       ├── pharmacist.py
│       └── radiologist.py
├── workspace/roles/          # 角色 prompt
│   ├── PHYSICIAN.md
│   ├── PHARMACIST.md
│   └── RADIOLOGIST.md
├── service/experiment.py     # 4 组实验框架
├── api/recommend.py          # 推荐气泡 API
└── eval/
    ├── data_generators/      # 5 个数据生成脚本
    ├── notebooks/            # 5 个训练 notebook
    ├── inference/            # 模型加载与集成
    └── models/               # 训练产物（gitignore）

frontend/
└── src/components/chat/
    ├── RoleOpinionCard.tsx   # 角色意见卡片
    └── RecommendBubbles.tsx  # 推荐气泡
```
