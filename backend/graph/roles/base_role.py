"""角色智能体基类 —— 每个角色复用同一 LLM，通过不同 system prompt 实现专业化。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from config import get_settings
from graph.llm import build_llm_config_from_settings, get_llm

logger = logging.getLogger(__name__)


class RoleType(str, Enum):
    PHYSICIAN = "physician"
    PHARMACIST = "pharmacist"
    RADIOLOGIST = "radiologist"


@dataclass
class RoleOpinion:
    """单个角色的诊断意见。"""

    role: RoleType
    role_label: str  # 中文角色名，用于前端展示
    content: str  # 角色回复正文
    evidence: list[str] = field(default_factory=list)  # 引用的证据来源
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class RoleAgent:
    """角色智能体基类。

    每个角色拥有独立的 system prompt，但共享同一个 LLM 实例。
    通过 system prompt 引导模型以特定专业角色的视角回答问题。
    """

    role_type: RoleType = RoleType.PHYSICIAN
    role_label: str = "主治医生"
    prompt_file: str = "PHYSICIAN.md"

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._system_prompt: str | None = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self._load_prompt()
        return self._system_prompt

    def _load_prompt(self) -> str:
        """从 workspace/roles/ 下加载角色 prompt 文件。"""
        prompt_path = self.base_dir / "workspace" / "roles" / self.prompt_file
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        logger.warning("Role prompt not found: %s, using default", prompt_path)
        return f"你是一名{self.role_label}，请从你的专业角度回答用户的医疗问题。"

    def _build_llm(self):
        settings = get_settings()
        llm_config = build_llm_config_from_settings(
            settings, temperature=0.3, streaming=False
        )
        return get_llm(llm_config)

    async def aconsult(
        self,
        query: str,
        context: str = "",
        memory_context: str = "",
    ) -> RoleOpinion:
        """异步会诊：给定用户问题和上下文，返回该角色的专业意见。

        Args:
            query: 用户原始问题
            context: 协调器提供的补充上下文（如其他角色的初步意见）
            memory_context: 共享记忆检索结果
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        # 注入共享记忆
        if memory_context:
            messages.append({
                "role": "system",
                "content": f"[共享长期记忆]\n{memory_context}",
            })

        # 注入协调器上下文
        if context:
            messages.append({
                "role": "system",
                "content": f"[会诊背景信息]\n{context}",
            })

        messages.append({"role": "user", "content": query})

        try:
            llm = self._build_llm()
            response = await llm.ainvoke(messages)
            content = self._extract_content(response)
        except Exception as exc:
            logger.error("Role %s consultation failed: %s", self.role_type.value, exc)
            content = f"[{self.role_label}会诊暂时不可用: {exc}]"

        return RoleOpinion(
            role=self.role_type,
            role_label=self.role_label,
            content=content,
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts).strip()
        return str(content or "").strip()
