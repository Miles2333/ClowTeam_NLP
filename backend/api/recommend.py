"""推荐气泡 API —— 首页基于近期会话生成可点击的推荐问题。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from config import get_settings
from graph.agent import agent_manager
from graph.llm import build_llm_config_from_settings, get_llm

logger = logging.getLogger(__name__)

router = APIRouter()

# 默认推荐（冷启动时使用）
DEFAULT_RECOMMENDATIONS = [
    {"id": "default_1", "text": "最近总是头痛、肩颈僵硬，需要做什么检查？"},
    {"id": "default_2", "text": "持续咳嗽两周没好，吃什么药合适？"},
    {"id": "default_3", "text": "体检报告显示肺部结节 5mm，需要进一步做CT吗？"},
    {"id": "default_4", "text": "高血压患者可以同时吃阿司匹林和布洛芬吗？"},
    {"id": "default_5", "text": "老年人腰痛拍片还是做MRI更合适？"},
]


def _extract_recent_topics(limit: int = 10) -> list[str]:
    """从近期会话中提取用户问题（只取 user role 的第一条消息）。"""
    if agent_manager.session_manager is None:
        return []

    sessions = agent_manager.session_manager.list_sessions()
    recent_queries: list[str] = []

    for session_summary in sessions[:limit]:
        session_id = session_summary.get("id")
        if not session_id:
            continue
        record = agent_manager.session_manager.load_session_record(session_id)
        for msg in record.get("messages", []):
            if msg.get("role") == "user":
                content = str(msg.get("content", "")).strip()
                if content:
                    recent_queries.append(content[:200])
                    break

    return recent_queries


async def _generate_recommendations(recent_queries: list[str]) -> list[dict[str, str]]:
    """基于近期问题让 LLM 生成 3-5 个推荐问题。"""
    if not recent_queries:
        return DEFAULT_RECOMMENDATIONS[:4]

    settings = get_settings()
    llm_config = build_llm_config_from_settings(settings, temperature=0.7, streaming=False)
    llm = get_llm(llm_config)

    queries_text = "\n".join(f"- {q}" for q in recent_queries[:10])

    prompt = (
        "你是 ClawTeam 医疗咨询系统的推荐助手。\n"
        "根据用户最近问过的问题，生成 4 个可能感兴趣的推荐问题。\n"
        "要求：\n"
        "1. 每个问题在 15-30 字之间\n"
        "2. 覆盖不同医疗场景（诊断、用药、检查、预防等）\n"
        "3. 贴近用户关注的主题但不要完全重复\n"
        "4. 使用日常口语，不要医学术语堆砌\n"
        "5. 只输出 4 个问题，每行一个，不要编号，不要解释。"
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"用户最近的问题：\n{queries_text}"},
        ])
        content = getattr(response, "content", "")
        text = content if isinstance(content, str) else str(content)

        lines = [line.strip().lstrip("-•0123456789. ").strip()
                 for line in text.splitlines() if line.strip()]
        lines = [l for l in lines if 5 < len(l) < 100][:5]

        if not lines:
            return DEFAULT_RECOMMENDATIONS[:4]

        return [{"id": f"rec_{i}", "text": text} for i, text in enumerate(lines)]
    except Exception as exc:
        logger.warning("Recommendation generation failed: %s", exc)
        return DEFAULT_RECOMMENDATIONS[:4]


@router.get("/recommend")
async def recommend() -> dict[str, Any]:
    """获取首页推荐问题气泡。"""
    recent_queries = _extract_recent_topics(limit=10)
    recommendations = await _generate_recommendations(recent_queries)
    return {"recommendations": recommendations}
