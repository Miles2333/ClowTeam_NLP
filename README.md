# ClawTeam

> Harness-based tumor board multi-agent consultation system
> 肿瘤多学科会诊（Tumor Board）系统：Harness Engineering 为主，Qwen3-4B LoRA 训练为辅。

ClawTeam 是一个面向肿瘤 MDT 场景的多智能体协作系统。项目重点不是单纯训练一个医疗模型，而是搭建一个可观测、可消融、可部署的 **Harness Engineering** 框架：通过复杂度评估、专科角色编排、多轮反驳修正、共识仲裁和可插拔 LoRA 专家，组织多个模型/角色完成肿瘤病例会诊。

## 项目定位

- **场景**：肿瘤多学科会诊（Tumor Board）
- **主线**：Harness Engineering（约 70%）
- **训练**：Qwen3-4B 外科 / 肿瘤内科 LoRA（约 30%）
- **目标**：给定肿瘤病例，输出包含手术、化疗、放疗、靶向/免疫、治疗时间线的综合 MDT 建议
- **说明**：本项目用于课程项目、实验评测和 Demo，不用于真实医疗决策

## 核心机制

| 机制 | 说明 |
|------|------|
| Complexity Assessor | 判断病例复杂度：simple / moderate / complex |
| Specialist Agents | 4 个肿瘤专科角色：病理、外科、肿瘤内科、放疗 |
| Round 1 Independent Reasoning | 各专科独立给出初始意见，不读取其他专家观点 |
| Round 2 Critique and Revision | 各专科阅读他人意见，强制输出同意、反对和修正 |
| Round 3 Consensus Arbitration | 协调器按专科相关性和分歧情况聚合最终方案 |
| LoRA Expert Plug-in | 外科和肿瘤内科可切换到云端训练好的 Qwen3-4B LoRA |
| Guardian | 可选安全守卫，用于高风险医疗请求拦截 |
| Ablation Harness | 通过 06 notebook 跑主实验和子消融 |

## MDT 角色

| 角色 | 实现方式 | 主要职责 |
|------|----------|----------|
| 病理科医生 | API + prompt | 组织学类型、TNM 分期、分子标志物、病理风险 |
| 肿瘤外科医生 | Qwen3-4B LoRA 或 API + prompt | 可切除性、术式、淋巴清扫、围手术期风险 |
| 肿瘤内科医生 | Qwen3-4B LoRA 或 API + prompt | 化疗、靶向、免疫、新辅助/辅助治疗 |
| 放疗科医生 | API + prompt | 放疗适应证、剂量分割、靶区和 OAR 保护 |

## 系统流程

```text
User Case
   |
   v
Guardian (optional)
   |
   v
Complexity Assessor
   |
   +-- simple   -> single specialist / direct answer
   |
   +-- moderate -> Round 1 four specialists -> aggregation
   |
   +-- complex  -> Round 1 -> Round 2 critique -> Round 3 arbitration
   |
   v
Final MDT Recommendation
```

## 消融实验设计

主实验保留 4 组：

| 组别 | 配置 | 验证问题 |
|------|------|----------|
| E1 | 单 Agent API | 基线能力 |
| E2 | 4 角色独立，无辩论，无 LoRA | 角色分工是否有用 |
| E3 | 4 角色 + 多轮辩论，无 LoRA | Harness 协作是否有用 |
| E4 | E3 + 外科/内科 LoRA | 训练专家是否带来增益 |

子消融建议：

- 外科 / 内科 LoRA vs API + prompt
- Complexity / Debate / Consensus / Guardian 逐项加入
- Round 数量：1 / 2 / 3 轮的效果与成本曲线

主要 notebook：

```text
backend/eval/notebooks/03_train_surgeon_qwen3.ipynb
backend/eval/notebooks/04_train_oncologist_qwen3.ipynb
backend/eval/notebooks/06_ablation_evaluation.ipynb
```

## 项目结构

```text
ClawTeam_NLP/
├── backend/
│   ├── api/                      # FastAPI 接口
│   ├── graph/
│   │   ├── agent.py              # 单 Agent / 多 Agent 流式入口
│   │   ├── coordinator.py        # MDT 协调器：Round 1/2/3
│   │   ├── complexity_assessor.py# 病例复杂度评估器
│   │   ├── guardian.py           # 安全守卫
│   │   └── roles/                # 4 个肿瘤专科 Agent
│   ├── workspace/
│   │   ├── AGENTS.md             # 当前 MDT 协作协议
│   │   └── roles/                # 角色 prompt
│   ├── eval/
│   │   ├── datasets/             # Benchmark 下载和筛选
│   │   ├── data_generators/      # 外科/内科训练数据准备
│   │   ├── inference/            # LoRA / Guardian 加载器
│   │   └── notebooks/            # 训练与消融实验
│   └── config/.env.example       # 后端配置模板
└── frontend/
    └── src/
        ├── components/chat/      # MDT 流程链、工具链、角色卡片
        ├── components/layout/    # 顶栏、会诊队列
        └── lib/                  # API 与状态管理
```

## 快速开始

### 1. 后端

```bash
cd backend
pip install -r requirements.txt
cp config/.env.example config/.env
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

本地测试时，如果没有云端 LoRA 模型，把 `.env` 中 LoRA 开关设为：

```env
USE_LORA_SURGEON=false
USE_LORA_MEDICAL_ONCOLOGIST=false
```

### 2. 前端

```bash
cd frontend
npm install
npm run dev
```

浏览器打开：

```text
http://localhost:3000
```

前端默认使用：

```text
http://<当前浏览器 hostname>:8002/api
```

## 云端 LoRA 配置

AutoDL 云端训练完成后，推荐在 `backend/config/.env` 中配置：

```env
USE_LORA_SURGEON=true
LORA_SURGEON_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_SURGEON_PATH=eval/models/surgeon_qwen3_lora

USE_LORA_MEDICAL_ONCOLOGIST=true
LORA_MEDICAL_ONCOLOGIST_BASE=/root/autodl-tmp/Qwen3-4B-Instruct-2507
LORA_MEDICAL_ONCOLOGIST_PATH=eval/models/oncologist_qwen3_lora
```

如果 adapter 不存在或本地没有 GPU，系统会回退到 API + prompt。

## 技术栈

### Backend

- Python
- FastAPI + SSE
- LangChain / LangGraph
- OpenAI-compatible LLM API
- PEFT / Transformers / Qwen3 LoRA
- Optional memory and Guardian modules inherited from miniOpenClaw

### Frontend

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- lucide-react

## 当前状态

已完成：

- 4 个肿瘤 MDT 角色 prompt 和 Agent 类
- Complexity Assessor
- Round 1 / Round 2 / Round 3 Harness 流程
- 前端 MDT 流程链、动态工具链、角色意见卡片
- DeepSeek OpenAI-compatible fallback
- Qwen3 外科 / 内科 LoRA notebook
- 06 消融实验 notebook

仍需进一步强化：

- 更严格的医学专用工具，如 TNM 表、药典、剂量计算器、指南检索
- 更细粒度的 Round 2 反驳质量评估
- 更完整的 Benchmark 结果与论文图表
- Guardian 在 06 消融中的完整接入验证

## 免责声明

ClawTeam 输出仅用于医学多智能体系统演示、课程项目和实验评测，不能替代真实医生的诊断、处方、手术或放疗计划。

## Acknowledgement

本项目基于 miniOpenClaw 改造，并保留其文件优先、技能文档、记忆模块和可审计 Agent 工作台思想。
