"""协调器（Coordinator）— v3.1 真协作 Harness 版

核心机制（参考 MDAgents NeurIPS 2024 + MDTeamGPT 2025）：

1. 复杂度评估 → 简单题单 agent / 中等题独立聚合 / 复杂题多轮辩论
2. Round 1：4 个角色并行独立思考（看不到他人）
3. Round 2：每个角色看到他人意见后，强制给出"同意/反对/修正"
4. Round 3：协调器加权聚合 + 主治仲裁

不是顺序流水线，而是真正的多视角碰撞。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import get_settings
from graph.complexity_assessor import (
    CaseComplexity,
    ComplexityDecision,
    assess_complexity,
)
from graph.llm import build_llm_config_from_settings, get_llm
from graph.roles.base_role import RoleAgent, RoleOpinion, RoleType
from graph.roles.medical_oncologist import MedicalOncologistAgent
from graph.roles.pathologist import PathologistAgent
from graph.roles.radiation_oncologist import RadiationOncologistAgent
from graph.roles.surgeon import SurgeonAgent

logger = logging.getLogger(__name__)


# ───────────── 角色权重（按问题相关性加权投票）─────────────

ROLE_RELEVANCE_KEYWORDS = {
    RoleType.PATHOLOGIST: ["分期", "TNM", "病理", "分化", "标志物", "EGFR", "ALK", "PD-L1"],
    RoleType.SURGEON: ["手术", "切除", "可切性", "术式", "淋巴清扫", "围手术期"],
    RoleType.MEDICAL_ONCOLOGIST: ["化疗", "靶向", "免疫", "新辅助", "辅助", "药物", "剂量"],
    RoleType.RADIATION_ONCOLOGIST: ["放疗", "Gy", "IMRT", "SBRT", "剂量", "靶区", "OAR"],
}


def compute_role_weights(case: str) -> dict[RoleType, float]:
    """根据 case 内容动态分配 4 角色权重（用于共识聚合）。"""
    weights = {}
    for role, keywords in ROLE_RELEVANCE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in case)
        # 基础权重 0.5，每命中关键词 +0.2，最多 1.5
        weights[role] = min(1.5, 0.5 + hits * 0.2)
    return weights


# ───────────── 数据类 ─────────────

@dataclass
class MDTSession:
    """一次 MDT 会诊的完整记录（用于评测和论文）。"""

    case: str
    complexity: ComplexityDecision | None = None
    round1_opinions: list[RoleOpinion] = field(default_factory=list)
    round2_opinions: list[RoleOpinion] = field(default_factory=list)
    final_decision: str = ""
    role_weights: dict[str, float] = field(default_factory=dict)
    revision_rate: float = 0.0  # Round 2 修正率（论文核心指标）
    disagreement_count: int = 0
    latency_ms: int = 0


# ───────────── Coordinator ─────────────

class Coordinator:
    """肿瘤 MDT 协调器 — 真协作 Harness 实现。

    根据复杂度选择不同协作模式：
    - SIMPLE   → 单 agent (默认 internist 角色，或 LLM 直接回答)
    - MODERATE → 4 角色 Round 1；完整模式继续 Round 2，消融时可跳过
    - COMPLEX  → 4 角色 Round 1 + Round 2 辩论 + Round 3 共识仲裁
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._roles: dict[RoleType, RoleAgent] = {
            RoleType.PATHOLOGIST: PathologistAgent(base_dir),
            RoleType.SURGEON: SurgeonAgent(base_dir),
            RoleType.MEDICAL_ONCOLOGIST: MedicalOncologistAgent(base_dir),
            RoleType.RADIATION_ONCOLOGIST: RadiationOncologistAgent(base_dir),
        }

    async def consult(
        self,
        case: str,
        *,
        memory_context: str = "",
        complexity_method: str = "llm",
        force_complexity: CaseComplexity | None = None,
        skip_round2: bool = False,  # 消融实验：关闭 Round 2 辩论
        attachments: list[dict[str, Any]] | None = None,
    ) -> MDTSession:
        """执行一次 MDT 会诊。

        Args:
            case: 病例描述
            memory_context: 历史病例参考（选做）
            complexity_method: "llm" / "bert" / "keyword"
            force_complexity: 实验时强制指定复杂度
            skip_round2: 消融实验时跳过辩论
        """
        start = time.monotonic()
        session = MDTSession(case=case)

        # ── Step 1: 复杂度评估 ──────────────────────
        if force_complexity is not None:
            session.complexity = ComplexityDecision(
                level=force_complexity,
                reason="手动指定（实验）",
                method="manual",
            )
        else:
            session.complexity = await assess_complexity(case, method=complexity_method)
        logger.info(
            "Complexity: %s (%s)",
            session.complexity.level.value,
            session.complexity.reason,
        )

        # ── Step 2: 根据复杂度选择策略 ──────────────
        if session.complexity.level == CaseComplexity.SIMPLE:
            # 单 agent 路径：用最相关的角色直接答
            session.final_decision = await self._simple_path(case, memory_context, attachments)
        else:
            # MDT 路径：4 角色协作
            session.role_weights = {
                k.value: v for k, v in compute_role_weights(case).items()
            }

            # ── Round 1: 并行独立思考 ───────────
            round1 = await self._run_round1(case, memory_context, attachments)
            session.round1_opinions = round1

            if skip_round2:
                # 消融实验 E2：直接聚合 Round 1，不跑辩论
                session.final_decision = await self._aggregate(
                    case, round1, weights=compute_role_weights(case)
                )
            else:
                # 完整 MDT：所有非 simple 病例进入 Round 2 辩论
                round2 = await self._run_round2(case, round1, memory_context, attachments)
                session.round2_opinions = round2

                # 计算论文核心指标：修正率
                session.revision_rate = self._compute_revision_rate(round1, round2)
                session.disagreement_count = sum(len(op.disagreements) for op in round2)

                # ── Round 3: 协调器仲裁 ──────────
                session.final_decision = await self._arbitrate(
                    case, round2, weights=compute_role_weights(case)
                )

        session.latency_ms = int((time.monotonic() - start) * 1000)
        return session

    # ───────────── 内部方法 ─────────────

    async def _simple_path(
        self,
        case: str,
        memory_context: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        """简单题：用最相关的角色（默认肿瘤内科）直接答。"""
        agent = self._roles[RoleType.MEDICAL_ONCOLOGIST]
        opinion = await agent.aconsult_round1(case, memory_context, attachments=attachments)
        return opinion.content

    async def _run_round1(
        self,
        case: str,
        memory_context: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> list[RoleOpinion]:
        """Round 1：4 角色并行独立思考。"""
        tasks = [
            agent.aconsult_round1(case, memory_context, attachments=attachments)
            for agent in self._roles.values()
        ]
        opinions = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [op for op in opinions if isinstance(op, RoleOpinion)]
        for op in opinions:
            if isinstance(op, Exception):
                logger.error("Round 1 role failed: %s", op)
        return valid

    async def _run_round2(
        self,
        case: str,
        round1: list[RoleOpinion],
        memory_context: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> list[RoleOpinion]:
        """Round 2：每个角色看完他人意见后，强制给出反驳/修正。

        ⭐ 这是真协作 Harness 的核心。
        """
        tasks = []
        for own in round1:
            others = [op for op in round1 if op.role != own.role]
            agent = self._roles.get(own.role)
            if agent is None:
                continue
            tasks.append(
                agent.aconsult_round2(
                    case, own, others, memory_context, attachments=attachments
                )
            )
        opinions = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [op for op in opinions if isinstance(op, RoleOpinion)]
        for op in opinions:
            if isinstance(op, Exception):
                logger.error("Round 2 role failed: %s", op)
        return valid

    @staticmethod
    def _compute_revision_rate(
        round1: list[RoleOpinion], round2: list[RoleOpinion]
    ) -> float:
        """计算 Round 2 修正率（论文核心指标）。

        定义：Round 2 中明确给出"修正"内容的角色占比。
        """
        if not round2:
            return 0.0
        revised = sum(1 for op in round2 if op.revisions and op.revisions[0] != "坚持 Round 1 判断")
        return revised / len(round2)

    async def _aggregate(
        self,
        case: str,
        opinions: list[RoleOpinion],
        weights: dict[RoleType, float],
    ) -> str:
        """中等复杂度的简单聚合（无辩论场景）。"""
        return await self._build_synthesis(
            case, opinions, weights, mode="aggregate"
        )

    async def _arbitrate(
        self,
        case: str,
        opinions: list[RoleOpinion],
        weights: dict[RoleType, float],
    ) -> str:
        """复杂度 COMPLEX 的最终仲裁（主治视角）。"""
        return await self._build_synthesis(
            case, opinions, weights, mode="arbitrate"
        )

    async def _build_synthesis(
        self,
        case: str,
        opinions: list[RoleOpinion],
        weights: dict[RoleType, float],
        mode: str,
    ) -> str:
        if not opinions:
            return "暂无可用专家意见。"

        # 拼接所有意见 + 权重
        opinion_text = "\n\n".join(
            f"【{op.role_label}（权重 {weights.get(op.role, 1.0):.1f}）】\n{op.content}"
            for op in opinions
        )

        if mode == "arbitrate":
            instruction = (
                "你是 Tumor Board 主任，请综合以下 4 位专家 Round 2 的意见做最终仲裁：\n"
                "1. 共识区域 → 直接采纳\n"
                "2. 冲突区域 → 按专家权重投票，权重高者优先\n"
                "3. 严重分歧 → 给出折中方案，并说明理由\n"
                "4. 输出综合治疗方案（含手术 / 化疗 / 放疗 / 时间线）\n"
                "5. 最后注明：以上建议仅供参考，具体诊疗请以实际就医为准"
            )
        else:
            instruction = (
                "你是 Tumor Board 协调员，请综合以下 4 位专家 Round 1 的独立意见：\n"
                "1. 整合各专科观点\n"
                "2. 按权重加权\n"
                "3. 输出综合治疗方案\n"
                "4. 末尾加免责声明"
            )

        settings = get_settings()
        llm = get_llm(
            build_llm_config_from_settings(settings, temperature=0.2, streaming=False)
        )

        try:
            response = await llm.ainvoke([
                {"role": "system", "content": instruction},
                {"role": "user", "content": f"【病例】\n{case}\n\n{opinion_text}"},
            ])
            content = getattr(response, "content", "")
            if isinstance(content, str):
                return content.strip()
            return str(content or "").strip()
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            # 降级：直接拼接
            parts = [f"**{op.role_label}**：{op.content[:200]}..." for op in opinions]
            return (
                "\n\n---\n\n".join(parts)
                + "\n\n> 以上仅供参考，具体诊疗请以实际就医为准。"
            )
