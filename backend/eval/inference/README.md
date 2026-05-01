# 推理加载工具

训练完成后，通过这些模块把训好的模型接入 ClawTeam 后端。**所有切换都通过环境变量，不需要改主代码。**

## 三个加载器

| 模块 | 替换 | 触发开关 |
|------|------|---------|
| `load_router.py` | `coordinator.route_by_keywords` | `USE_TRAINED_ROUTER=true` |
| `load_guardian.py` | `graph/guardian.py` 的 prompt 调用 | `USE_TRAINED_GUARDIAN=true` |
| `load_lora_role.py` | `graph/roles/base_role.py` 的 LLM API | `USE_LORA_{ROLE}=true` |

## 完整 .env 示例

训完所有模型后，在 `backend/config/.env` 加上：

```env
# === 训练好的小模型（BERT）===
USE_TRAINED_ROUTER=true
TRAINED_ROUTER_PATH=eval/models/router_bert

USE_TRAINED_GUARDIAN=true
TRAINED_GUARDIAN_PATH=eval/models/guardian_bert

# === LoRA 角色（每个角色独立开关）===
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

## 如何接入主代码（最小侵入式）

### 1. Coordinator 加路由切换

修改 `backend/graph/coordinator.py` 在 `consult` 方法开头加：

```python
async def consult(self, query: str, **kwargs):
    # ===== 优先使用训练好的路由器 =====
    from eval.inference.load_router import load_trained_router
    trained_router = load_trained_router()
    if trained_router is not None and 'enabled_roles' not in kwargs:
        decisions = trained_router.route(query)
        from graph.roles.base_role import RoleType
        kwargs['enabled_roles'] = [
            RoleType.PHYSICIAN if decisions['physician'] else None,
            RoleType.PHARMACIST if decisions['pharmacist'] else None,
            RoleType.RADIOLOGIST if decisions['radiologist'] else None,
        ]
        kwargs['enabled_roles'] = [r for r in kwargs['enabled_roles'] if r]
    # ===== 否则回退到关键词路由（默认） =====
    
    # ... 后续原有逻辑不变 ...
```

### 2. Guardian 加分类器切换

修改 `backend/graph/guardian.py` 的 `evaluate_guardian_input`：

```python
def evaluate_guardian_input(user_text: str) -> GuardianRuntimeResult:
    # ===== 优先使用训练好的 BERT Guardian =====
    from eval.inference.load_guardian import load_trained_guardian
    trained = load_trained_guardian()
    if trained is not None:
        label, confidence = trained.classify(user_text)
        is_safe = (label == "safe")
        return GuardianRuntimeResult(
            is_blocked=not is_safe,
            label="安全" if is_safe else "危险",
            reason_code=f"trained_{label}",
            block_message=get_settings().guardian_block_message,
        )
    # ===== 否则回退到 prompt-based Guardian =====
    
    # ... 原有逻辑不变 ...
```

### 3. 角色 Agent 加 LoRA 切换

修改 `backend/graph/roles/base_role.py` 的 `aconsult` 方法：

```python
async def aconsult(self, query: str, context: str = "", memory_context: str = ""):
    # ===== 优先使用 LoRA 微调模型 =====
    from eval.inference.load_lora_role import load_lora_role
    lora_agent = load_lora_role(self.role_type.value)
    if lora_agent is not None:
        try:
            content = lora_agent.generate(
                system_prompt=self.system_prompt,
                user_text=query,
                max_new_tokens=512,
                temperature=0.3,
            )
            return RoleOpinion(
                role=self.role_type,
                role_label=self.role_label,
                content=content,
            )
        except Exception as exc:
            logger.warning("LoRA inference failed for %s, falling back to API: %s", 
                         self.role_type.value, exc)
    # ===== 否则回退到 LLM API（默认） =====
    
    # ... 原有逻辑不变 ...
```

## 切换实验对照（论文用）

| 实验场景 | 配置 |
|---------|------|
| 纯 API 基线 | 所有 `USE_*` 设为 false |
| 仅小模型替换 | `USE_TRAINED_ROUTER/GUARDIAN=true`，LoRA 关 |
| 全 LoRA 替换 | 所有 `USE_*` 设为 true |
| 消融某一项 | 单独关掉对应 `USE_*` |

每次切换只改 `.env` + 重启后端，便于跑论文对比实验。

## 显存预估（5070 Ti 12GB）

| 配置 | 显存占用 |
|------|---------|
| 只 BERT (路由 + Guardian) | ~1 GB |
| + 主治 + 药师 LoRA (0.5B 各一份) | ~5 GB |
| + 影像科 VL LoRA (2B + 4-bit) | ~9 GB |
| **全部启用** | **约 9-11 GB** ✅ 能跑 |

如果显存不够：
- 把 `USE_LORA_RADIOLOGIST=false`（多模态最吃显存）
- 或把基座换成更小的（如 Qwen2-VL-2B 已经是最小了，无法更小）
