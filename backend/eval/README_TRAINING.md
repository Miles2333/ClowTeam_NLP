# ClawTeam 训练指南

本目录提供 ClawTeam 系统的 5 个训练任务，按本地小模型版（笔记本可跑）实现。

## 训练目标

| 编号 | 任务 | 模型 | 显存 | 时间 |
|------|------|------|------|------|
| 01 | 路由分类器 | bert-base-chinese (110M) | 2-3 GB | 20 分钟 |
| 02 | Guardian 安全守卫 | bert-base-chinese (110M) | 2-3 GB | 20 分钟 |
| 03 | 主治医生 LoRA | Qwen2.5-0.5B-Instruct | 4-6 GB | 1-2 小时 |
| 04 | 临床药师 LoRA | Qwen2.5-0.5B-Instruct | 4-6 GB | 1-2 小时 |
| 05 | 影像科多模态 LoRA | Qwen2-VL-2B-Instruct | 10-12 GB | 3-4 小时 |

## 环境准备

### 1. 检查 GPU
```bash
nvidia-smi
```
确认 CUDA Version >= 12.4（5070 Ti 需要这个版本）

### 2. 创建独立环境（强烈推荐）
```bash
conda create -n clawteam-train python=3.10 -y
conda activate clawteam-train
```

### 3. 安装依赖
```bash
cd D:\NLP_Project\miniOpenClaw\backend\eval
pip install -r requirements_train.txt

# 如果 PyTorch 默认装的不是 CUDA 12.4 版本，重装：
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```

### 4. 验证 GPU 可用
```python
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('设备:', torch.cuda.get_device_name(0))"
```
应该输出：
```
CUDA可用: True
设备: NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

### 5. 配置 API Key（用于自动生成数据）

在 `backend/config/.env` 中确认有 DeepSeek 或类似 API Key（你 MVP 时配过的）：
```env
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

数据生成脚本会读这个配置自动生成训练数据。

### 6. （可选）注册 wandb 训练追踪
```bash
pip install wandb
wandb login   # 跟提示走，粘贴 token
```
不想用就在 notebook 里设 `os.environ["WANDB_DISABLED"] = "true"`

---

## 完整执行流程

### 阶段 1：生成训练数据（不需要 GPU）

```bash
cd D:\NLP_Project\miniOpenClaw\backend\eval

# 路由分类数据（约 30 分钟，调 1000 次 API）
python data_generators/gen_router_data.py

# Guardian 攻击集 + 正常集（约 30 分钟）
python data_generators/gen_guardian_data.py

# 主治问答对（约 1 小时）
python data_generators/gen_physician_data.py

# 药师问答对（约 1 小时）
python data_generators/gen_pharmacist_data.py

# 影像描述（约 1 小时）
python data_generators/gen_radiologist_data.py
```

每个脚本生成 500-1000 条数据到 `datasets/data/`。脚本支持中断恢复。

**API 成本估算**：DeepSeek 调用约 5000 次，总计约 ¥10-30。

### 阶段 2：训练（用 GPU）

```bash
jupyter lab
# 浏览器打开后，进入 notebooks/，逐个运行：
```

**推荐顺序**（从简单到复杂）：

1. `01_train_router.ipynb` ← 先跑这个建立信心
2. `02_train_guardian.ipynb`
3. `03_train_physician_lora.ipynb`
4. `04_train_pharmacist_lora.ipynb`
5. `05_train_radiologist_vl.ipynb`（最难，最后跑）

每个 notebook 都是**逐 cell 运行**：
- 第一个 cell：环境检查（确认 GPU 可用）
- 中间 cell：加载数据、模型、训练
- 最后 cell：保存模型 + 推理测试

训练完成后，模型自动保存到 `eval/models/{任务名}/`。

### 阶段 3：集成回 ClawTeam 后端

训完后，修改 `backend/config/.env`：

```env
# 启用训练好的小模型
USE_TRAINED_ROUTER=true
TRAINED_ROUTER_PATH=eval/models/router_bert

USE_TRAINED_GUARDIAN=true
TRAINED_GUARDIAN_PATH=eval/models/guardian_bert

# LoRA 角色（按需启用）
USE_LORA_PHYSICIAN=true
LORA_PHYSICIAN_BASE=Qwen/Qwen2.5-0.5B-Instruct
LORA_PHYSICIAN_PATH=eval/models/physician_lora

USE_LORA_PHARMACIST=true
LORA_PHARMACIST_BASE=Qwen/Qwen2.5-0.5B-Instruct
LORA_PHARMACIST_PATH=eval/models/pharmacist_lora

USE_LORA_RADIOLOGIST=true
LORA_RADIOLOGIST_BASE=Qwen/Qwen2-VL-2B-Instruct
LORA_RADIOLOGIST_PATH=eval/models/radiologist_vl_lora
```

然后正常启动后端：
```bash
cd backend
uvicorn app:app --port 8002 --reload
```

后端会自动检测配置，加载训练好的模型替换 prompt-only 角色。

---

## 常见问题

### Q1: CUDA out of memory
**原因**：显存不够。
**解决**：
- LoRA notebook 里把 `per_device_train_batch_size` 从 4 改成 2 或 1
- 启用 4-bit 量化：`load_in_4bit=True`
- BERT 训练把 `batch_size` 从 16 改成 8

### Q2: 数据生成脚本报错 "Connection error"
**原因**：API Key 没配好，或网络问题。
**解决**：检查 `backend/config/.env`，再跑 `python -c "from openai import OpenAI; ..."` 测试 API 联通。

### Q3: 训练 loss 不降
**原因**：学习率不对、数据质量差。
**解决**：
- Notebook 里调 `learning_rate`（默认 2e-5，可试 1e-5 到 5e-5）
- 检查生成数据质量（先看 `datasets/data/*.jsonl` 前 10 条）

### Q4: 训完后端加载不了模型
**原因**：路径配置错或模型文件缺失。
**解决**：
- 确认 `eval/models/router_bert/config.json` 存在
- 确认 `.env` 里 `TRAINED_ROUTER_PATH` 是相对 backend 目录的路径

---

## 训练完成后的下一步

本地小模型跑通后，下一步是 **云端训练大模型**（Qwen2.5-3B/7B + Qwen2.5-VL-3B/7B）。
那时只需要：
1. 把这套 notebook 拷贝到 AutoDL
2. 改 notebook 顶部的 `MODEL_NAME` 变量
3. 重新跑

代码逻辑完全不用改，只是换模型名字。
