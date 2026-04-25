"""协调器模块 —— 接收用户问题，路由到角色智能体，融合输出。

协调器是 ClawTeam 多智能体系统的核心枢纽：
1. 接收用户问题
2. 通过路由器判断需要调用哪些角色
3. 并行调用被选中的角色智能体
4. 融合各角色意见为最终综合结论
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import get_settings
from graph.llm import build_llm_config_from_settings, get_llm
from graph.roles.base_role import RoleAgent, RoleOpinion, RoleType
from graph.roles.physician import PhysicianAgent
from graph.roles.pharmacist import PharmacistAgent
from graph.roles.radiologist import RadiologistAgent

logger = logging.getLogger(__name__)

# ── 路由关键词规则（轻量级，后期可替换为训练好的分类器） ──────────────

_PHARMACIST_KEYWORDS = [
    "药", "用药", "剂量", "服药", "吃药", "药物", "处方", "抗生素",
    "止痛药", "消炎药", "副作用", "不良反应", "过敏", "禁忌",
    "相互作用", "DDI", "头孢", "阿莫西林", "布洛芬", "对乙酰氨基酚",
    "药片", "胶囊", "输液", "注射", "医嘱", "停药", "换药", "减量",
    "drug", "medication", "dose", "prescription", "antibiotic",
]

_RADIOLOGIST_KEYWORDS = [
    "CT", "MRI", "X光", "X线", "B超", "超声", "核磁", "造影",
    "拍片", "影像", "片子", "胸片", "腹部CT", "头颅MRI",
    "结节", "肿块", "阴影", "钙化", "占位", "肺结节",
    "BI-RADS", "Lung-RADS", "检查报告", "影像报告",
    "radiograph", "scan", "imaging", "ultrasound",
]


@dataclass
class RoutingDecision:
    """路由决策结果。"""

    need_physician: bool = True  # 主治医生默认必调
    need_pharmacist: bool = False
    need_radiologist: bool = False
    reason: str = ""

    @property
    def roles_needed(self) -> list[RoleType]:
        roles = [RoleType.PHYSICIAN]
        if self.need_pharmacist:
            roles.append(RoleType.PHARMACIST)
        if self.need_radiologist:
            roles.append(RoleType.RADIOLOGIST)
        return roles


@dataclass
class CoordinatorResult:
    """协调器输出结果。"""

    opinions: list[RoleOpinion] = field(default_factory=list)
    synthesis: str = ""  # 最终融合结论
    routing: RoutingDecision | None = None
    latency_ms: int = 0
    total_tokens: int = 0


def route_by_keywords(query: str) -> RoutingDecision:
    """基于关键词的规则路由（MVP 阶段）。

    后期可替换为训练好的分类器模型。
    """
    query_lower = query.lower()

    need_pharmacist = any(kw.lower() in query_lower for kw in _PHARMACIST_KEYWORDS)
    need_radiologist = any(kw.lower() in query_lower for kw in _RADIOLOGIST_KEYWORDS)

    reasons = ["主治医生(默认)"]
    if need_pharmacist:
        reasons.append("药师(关键词命中)")
    if need_radiologist:
        reasons.append("影像科(关键词命中)")

    return RoutingDecision(
        need_physician=True,
        need_pharmacist=need_pharmacist,
        need_radiologist=need_radiologist,
        reason=" + ".join(reasons),
    )


async def route_by_llm(query: str) -> RoutingDecision:
    """基于 LLM 的路由（可选，更准确但增加一次 LLM 调用延迟）。

    输出格式：physician,pharmacist,radiologist（逗号分隔的角色列表）
    """
    settings = get_settings()
    llm_config = build_llm_config_from_settings(settings, temperature=0.0, streaming=False)
    llm = get_llm(llm_config)

    prompt = (
        "你是一个医疗问题路由器。根据用户问题判断需要哪些专科医生参与回答。\n"
        "可选角色：physician（主治医生）、pharmacist（药师）、radiologist（影像科）\n"
        "规则：\n"
        "- physician 必须参与\n"
        "- 涉及用药、剂量、药物相互作用等问题时加入 pharmacist\n"
        "- 涉及影像检查、CT/MRI/X光/超声等问题时加入 radiologist\n"
        "只输出角色列表，逗号分隔，不要解释。\n"
        "示例输出：physician,pharmacist"
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ])
        content = getattr(response, "content", "").strip().lower()

        need_pharmacist = "pharmacist" in content
        need_radiologist = "radiologist" in content

        return RoutingDecision(
            need_physician=True,
            need_pharmacist=need_pharmacist,
            need_radiologist=need_radiologist,
            reason=f"LLM路由: {content}",
        )
    except Exception as exc:
        logger.warning("LLM routing failed, falling back to keyword: %s", exc)
        return route_by_keywords(query)


class Coordinator:
    """多角色协调器。

    负责：
    1. 路由决策（决定调用哪些角色）
    2. 并行调用角色智能体
    3. 融合多角色意见为最终结论
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._roles: dict[RoleType, RoleAgent] = {
            RoleType.PHYSICIAN: PhysicianAgent(base_dir),
            RoleType.PHARMACIST: PharmacistAgent(base_dir),
            RoleType.RADIOLOGIST: RadiologistAgent(base_dir),
        }

    async def consult(
        self,
        query: str,
        *,
        memory_context: str = "",
        use_llm_router: bool = False,
        enabled_roles: list[RoleType] | None = None,
    ) -> CoordinatorResult:
        """执行一次多角色会诊。

        Args:
            query: 用户原始问题
            memory_context: 共享记忆检索结果
            use_llm_router: 是否使用 LLM 路由（否则用关键词规则）
            enabled_roles: 强制指定角色列表（覆盖路由决策，用于实验）
        """
        start_time = time.monotonic()

        # 1. 路由决策
        if enabled_roles is not None:
            routing = RoutingDecision(
                need_physician=RoleType.PHYSICIAN in enabled_roles,
                need_pharmacist=RoleType.PHARMACIST in enabled_roles,
                need_radiologist=RoleType.RADIOLOGIST in enabled_roles,
                reason="手动指定",
            )
        elif use_llm_router:
            routing = await route_by_llm(query)
        else:
            routing = route_by_keywords(query)

        logger.info("Routing decision: %s (reason: %s)", routing.roles_needed, routing.reason)

        # 2. 并行调用角色智能体
        tasks = []
        for role_type in routing.roles_needed:
            agent = self._roles[role_type]
            tasks.append(agent.aconsult(
                query=query,
                memory_context=memory_context,
            ))

        opinions = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤掉异常结果
        valid_opinions: list[RoleOpinion] = []
        for opinion in opinions:
            if isinstance(opinion, Exception):
                logger.error("Role consultation failed: %s", opinion)
            else:
                valid_opinions.append(opinion)

        # 3. 融合多角色意见
        synthesis = await self._synthesize(query, valid_opinions)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return CoordinatorResult(
            opinions=valid_opinions,
            synthesis=synthesis,
            routing=routing,
            latency_ms=elapsed_ms,
        )

    async def _synthesize(
        self, query: str, opinions: list[RoleOpinion]
    ) -> str:
        """融合多角色意见为最终综合结论。"""
        if not opinions:
            return "暂无可用角色意见。"

        if len(opinions) == 1:
            # 只有主治医生，直接返回其意见
            return opinions[0].content

        # 多角色意见需要 LLM 融合
        settings = get_settings()
        llm_config = build_llm_config_from_settings(
            settings, temperature=0.2, streaming=False
        )
        llm = get_llm(llm_config)

        opinion_text = "\n\n".join(
            f"【{op.role_label}意见】\n{op.content}" for op in opinions
        )

        synthesis_prompt = (
            "你是 ClawTeam 多智能体协作诊疗系统的协调器。\n"
            "请综合以下各专科角色的意见，给出一份完整、连贯的诊疗建议。\n\n"
            "要求：\n"
            "1. 整合各角色意见，消除重复，解决矛盾\n"
            "2. 按照\"综合诊断意见 → 建议检查 → 治疗方向 → 注意事项\"的结构组织\n"
            "3. 标注各建议的来源角色（如\"主治医生建议...\"、\"药师提醒...\"）\n"
            "4. 如果角色意见有冲突，优先采纳主治医生意见并注明争议点\n"
            "5. 末尾必须加上免责声明：\"以上建议仅供参考，具体诊疗请以实际就医为准。\"\n"
        )

        try:
            response = await llm.ainvoke([
                {"role": "system", "content": synthesis_prompt},
                {"role": "user", "content": f"用户问题：{query}\n\n{opinion_text}"},
            ])
            content = getattr(response, "content", "")
            if isinstance(content, str):
                return content.strip()
            return str(content or "").strip()
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            # 降级：简单拼接
            parts = [f"**{op.role_label}**：{op.content}" for op in opinions]
            return "\n\n---\n\n".join(parts) + "\n\n> 以上建议仅供参考，具体诊疗请以实际就医为准。"
