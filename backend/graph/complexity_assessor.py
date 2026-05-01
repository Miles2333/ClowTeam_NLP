"""复杂度评估器（Complexity Assessor）— 不是 Router！

按 TA 反馈：Router 是为了省成本（大小模型池），我们这里不是 Router，
是判断病例复杂度，决定是否需要触发多轮 MDT 辩论。

参考 MDAgents (NeurIPS 2024) 的 Medical Complexity Check 设计：
- simple   → 单 agent 直接答（如基础肿瘤知识题，省 token）
- moderate → 4 角色独立给意见 + 简单聚合
- complex  → 4 角色 + 多轮辩论 + 仲裁（最重，但效果最好）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from config import get_settings
from graph.llm import build_llm_config_from_settings, get_llm

logger = logging.getLogger(__name__)


class CaseComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ComplexityDecision:
    level: CaseComplexity
    reason: str
    method: str  # "llm" | "bert" | "keyword"


# ───────────── 关键词规则（baseline） ─────────────

_COMPLEX_KEYWORDS = [
    "鉴别诊断", "MDT", "多学科", "晚期", "转移", "复发",
    "T3", "T4", "N2", "N3", "M1",
    "新辅助", "辅助治疗", "靶向", "免疫",
    "肝肾功能", "合并症", "高龄", "妊娠",
]

_SIMPLE_KEYWORDS = [
    "什么是", "定义", "请解释", "病理类型",
    "TNM 分期标准", "WHO 分类",
]


def assess_by_keyword(case: str) -> ComplexityDecision:
    """关键词规则评估（baseline）。"""
    case_lower = case.lower()
    hits_complex = sum(1 for kw in _COMPLEX_KEYWORDS if kw in case)
    hits_simple = sum(1 for kw in _SIMPLE_KEYWORDS if kw in case)

    if hits_simple > 0 and hits_complex == 0:
        level = CaseComplexity.SIMPLE
        reason = "关键词规则：检测到知识查询类问题"
    elif hits_complex >= 2:
        level = CaseComplexity.COMPLEX
        reason = f"关键词规则：检测到 {hits_complex} 个复杂决策关键词"
    else:
        level = CaseComplexity.MODERATE
        reason = "关键词规则：常规病例"

    return ComplexityDecision(level=level, reason=reason, method="keyword")


async def assess_by_llm(case: str) -> ComplexityDecision:
    """LLM 评估病例复杂度（默认方法）。"""
    settings = get_settings()
    llm_config = build_llm_config_from_settings(
        settings, temperature=0.0, streaming=False
    )
    llm = get_llm(llm_config)

    prompt = (
        "你是肿瘤多学科会诊系统的病例复杂度评估员。\n"
        "根据以下病例描述，判断其复杂度等级：\n"
        "- simple：基础知识问答（如定义、TNM 标准），单个专家可解决\n"
        "- moderate：常规肿瘤病例，4 个专科独立给意见即可\n"
        "- complex：疑难病例（多脏器累及、晚期、合并症、罕见类型），需要专家多轮辩论\n"
        "\n"
        "只输出一个词：simple / moderate / complex。"
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": case},
        ])
        content = getattr(response, "content", "").strip().lower()

        if "simple" in content:
            level = CaseComplexity.SIMPLE
        elif "complex" in content:
            level = CaseComplexity.COMPLEX
        else:
            level = CaseComplexity.MODERATE

        return ComplexityDecision(
            level=level, reason=f"LLM 评估：{content}", method="llm"
        )
    except Exception as exc:
        logger.warning("LLM complexity assessment failed: %s, falling back to keyword", exc)
        return assess_by_keyword(case)


def assess_by_bert(case: str) -> ComplexityDecision | None:
    """用之前训练好的 BERT Router 模型评估（已训完，作为对比组）。"""
    try:
        from eval.inference.load_router import load_trained_router
        router = load_trained_router()
        if router is None:
            return None

        # 之前的 BERT Router 是多标签分类（physician/pharmacist/radiologist）
        # 我们把"调用了几个角色"映射到复杂度
        probs = router.predict(case)
        active = sum(1 for p in probs.values() if p > 0.5)

        if active <= 1:
            level = CaseComplexity.SIMPLE
        elif active == 2:
            level = CaseComplexity.MODERATE
        else:
            level = CaseComplexity.COMPLEX

        return ComplexityDecision(
            level=level,
            reason=f"BERT 评估：检测到 {active} 个相关专科",
            method="bert",
        )
    except Exception as exc:
        logger.debug("BERT assessor not available: %s", exc)
        return None


async def assess_complexity(
    case: str,
    method: Literal["llm", "bert", "keyword"] = "llm",
) -> ComplexityDecision:
    """统一入口。"""
    if method == "bert":
        result = assess_by_bert(case)
        if result is not None:
            return result
        # 回退到 keyword
        return assess_by_keyword(case)
    if method == "keyword":
        return assess_by_keyword(case)
    return await assess_by_llm(case)
