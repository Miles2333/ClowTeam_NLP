from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logging

from config import get_settings, runtime_config
from graph.context import RequestContext
from graph.agent_factory import build_agent_config, create_agent_from_config
from graph.coordinator import Coordinator
from service.memory_indexer import memory_indexer
from service.session_manager import SessionManager
from service.experiment import ExperimentMode, ExperimentLog, experiment_logger
from graph.llm import build_llm_config_from_settings, get_llm
from tools import get_all_tools
from memory_module_v2.service.config import get_memory_backend, get_memory_v2_inject_mode
from memory_module_v2.integrations.middleware import build_memory_context

logger = logging.getLogger(__name__)


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


class AgentManager:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self.session_manager: SessionManager | None = None
        self.tools = []
        self.coordinator: Coordinator | None = None

    def initialize(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.session_manager = SessionManager(base_dir)
        self.tools = get_all_tools(base_dir)
        self.coordinator = Coordinator(base_dir)
        experiment_logger.configure(base_dir)

    # 用于generate_title()和summarize_history()
    def _build_chat_model(self):
        settings = get_settings()
        llm_config = build_llm_config_from_settings(settings, temperature=0.0, streaming=False)
        return get_llm(llm_config)

    def _build_agent(self):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")
        config = build_agent_config(
            self.base_dir, self.tools, use_checkpointer=True
        )
        return create_agent_from_config(config)

    def _build_messages(self, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in history:
            role = item.get("role")
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": str(item.get("content", ""))})
        return messages

    def _format_retrieval_context(self, results: list[dict[str, Any]]) -> str:
        lines = ["[RAG retrieved memory context]"]
        for idx, item in enumerate(results, start=1):
            text = str(item.get("text", "")).strip()
            source = str(
                item.get(
                    "source",
                    "memory_module_v1/long_term_memory/MEMORY.md",
                )
            )
            lines.append(f"{idx}. Source: {source}\n{text}")
        return "\n\n".join(lines)

    async def astream(
        self,
        message: str,
        history: list[dict[str, Any]],
        context: RequestContext | None = None,
    ):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")

        memory_backend = get_memory_backend()
        turn_messages: list[dict[str, str]] = []

        if memory_backend == "v1":
            # v1: Chroma / MEMORY.md RAG injection
            retrievals = memory_indexer.retrieve(message, top_k=3)
            yield {"type": "retrieval", "query": message, "results": retrievals}
            if retrievals:
                turn_messages.append(
                    {
                        "role": "assistant",
                        "content": self._format_retrieval_context(retrievals),
                    }
                )
        elif memory_backend == "v2" and get_memory_v2_inject_mode() == "always":
            # v2 forced injection: search every turn, prepend to prompt
            try:
                v2_context = build_memory_context(message)
                if v2_context:
                    yield {"type": "retrieval_v2", "query": message, "context": v2_context}
                    turn_messages.append(
                        {"role": "assistant", "content": v2_context}
                    )
            except Exception as v2_exc:
                logger.warning("Memory v2 forced injection failed: %s", v2_exc)
        # When memory_backend == "v2" and inject_mode == "tool":
        #   search_memory is registered as a tool in get_all_tools(),
        #   the agent decides when to call it autonomously.

        turn_messages.append({"role": "user", "content": message})

        agent = self._build_agent()
        run_config: dict[str, Any] = {"configurable": {"thread_id": (context.thread_id if context else "")}}
        if context and context.callbacks:
            run_config["callbacks"] = context.callbacks
        if not run_config["configurable"]["thread_id"]:
            run_config["configurable"]["thread_id"] = "default"

        final_content_parts: list[str] = []
        last_ai_message = ""
        pending_tools: dict[str, dict[str, str]] = {}
        last_usage: dict[str, Any] | None = None

        async for mode, payload in agent.astream(
            {"messages": turn_messages},
            stream_mode=["messages", "updates"],
            config=run_config,
            # stream_options={"include_usage": True}
        ):
            if mode == "messages":
                chunk, metadata = payload
                # 优先从 metadata 中读取 usage（LangGraph 在 include_usage=True 时会放在这里）
                usage_candidate: Any = None
                if isinstance(metadata, dict):
                    usage_candidate = metadata.get("usage")
                if isinstance(usage_candidate, dict):
                    last_usage = usage_candidate

                # 只转发主 agent 节点的 token；跳过 guardian middleware 等非 agent 节点的 LLM 输出
                node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
                if node is not None and node != "agent":
                    continue

                text = _stringify_content(getattr(chunk, "content", ""))
                if text:
                    final_content_parts.append(text)
                    yield {"type": "token", "content": text}
                continue

            if mode != "updates":
                continue

            for update in payload.values():
                if not update:
                    continue
                for agent_message in update.get("messages", []):
                    message_type = getattr(agent_message, "type", "")
                    tool_calls = getattr(agent_message, "tool_calls", []) or []

                    if message_type == "ai" and not tool_calls:
                        candidate = _stringify_content(getattr(agent_message, "content", ""))
                        if candidate:
                            last_ai_message = candidate

                    if tool_calls:
                        for tool_call in tool_calls:
                            call_id = str(tool_call.get("id") or tool_call.get("name"))
                            tool_name = str(tool_call.get("name", "tool"))
                            tool_args = tool_call.get("args", "")
                            if not isinstance(tool_args, str):
                                tool_args = json.dumps(tool_args, ensure_ascii=False)
                            pending_tools[call_id] = {
                                "tool": tool_name,
                                "input": str(tool_args),
                            }
                            yield {
                                "type": "tool_start",
                                "tool": tool_name,
                                "input": str(tool_args),
                            }

                    if message_type == "tool":
                        tool_call_id = str(getattr(agent_message, "tool_call_id", ""))
                        pending = pending_tools.pop(
                            tool_call_id,
                            {"tool": getattr(agent_message, "name", "tool"), "input": ""},
                        )
                        output = _stringify_content(getattr(agent_message, "content", ""))
                        yield {
                            "type": "tool_end",
                            "tool": pending["tool"],
                            "output": output,
                        }
                        yield {"type": "new_response"}

        final_content = "".join(final_content_parts).strip() or last_ai_message.strip()
        # 若 LLM 返回了 usage，且本次调用启用了 Langfuse，则在结束时补充 usage 信息，方便在 Langfuse 中显示 tokens
        if last_usage and context and context.callbacks:
            try:
                from langfuse import get_client
                from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

                langfuse_handler: Any | None = None
                for cb in context.callbacks:
                    if isinstance(cb, LangfuseCallbackHandler):
                        langfuse_handler = cb
                        break
                trace_id = getattr(langfuse_handler, "last_trace_id", None) if langfuse_handler else None
                if trace_id:
                    client = get_client()
                    client.trace.update(
                        id=trace_id,
                        usage={
                            "input": last_usage.get("prompt_tokens", 0),
                            "output": last_usage.get("completion_tokens", 0),
                            "total": last_usage.get("total_tokens", 0),
                        },
                    )
            except Exception as exc:
                print("[langfuse] 更新 usage 失败：", repr(exc))
        yield {"type": "done", "content": final_content}

    async def astream_multi_agent(
        self,
        message: str,
        history: list[dict[str, Any]],
        context: RequestContext | None = None,
        experiment_mode: ExperimentMode = ExperimentMode.MULTI_FULL,
    ):
        """多 Agent 模式流式输出（ClawTeam 协作诊疗）。

        事件类型：
        - role_opinion: 单个角色的诊断意见
        - synthesis_token: 融合结论的流式 token
        - retrieval_v2: 共享记忆检索结果
        - done: 最终完整结论
        """
        if self.base_dir is None or self.coordinator is None:
            raise RuntimeError("AgentManager is not initialized")

        # Guardian 前置检查（仅 MULTI_FULL 模式启用）
        if experiment_mode.use_guardian:
            from graph.guardian import evaluate_guardian_input
            guardian_result = evaluate_guardian_input(message)
            if guardian_result.is_blocked:
                yield {
                    "type": "guardian_blocked",
                    "label": guardian_result.label,
                    "reason": guardian_result.reason_code,
                    "message": guardian_result.block_message,
                }
                yield {"type": "done", "content": guardian_result.block_message}
                return

        # 共享记忆检索
        memory_context = ""
        if experiment_mode.use_shared_memory and get_memory_backend() == "v2":
            try:
                memory_context = build_memory_context(message) or ""
                if memory_context:
                    yield {
                        "type": "retrieval_v2",
                        "query": message,
                        "context": memory_context,
                    }
            except Exception as exc:
                logger.warning("Memory v2 retrieval failed in multi-agent mode: %s", exc)
        elif experiment_mode.use_shared_memory and get_memory_backend() == "v1":
            retrievals = memory_indexer.retrieve(message, top_k=3)
            if retrievals:
                yield {"type": "retrieval", "query": message, "results": retrievals}
                memory_context = self._format_retrieval_context(retrievals)

        # 多角色会诊（v3.1 真协作 Harness：Round 1 → Round 2 → Round 3）
        # 消融实验：根据 experiment_mode 决定是否跳过 Round 2
        skip_round2 = experiment_mode.value in ("multi_no_memory",)  # E2 不走辩论
        session = await self.coordinator.consult(
            case=message,
            memory_context=memory_context,
            skip_round2=skip_round2,
        )

        # 发送复杂度评估结果
        if session.complexity:
            yield {
                "type": "routing",  # 沿用前端事件名，意义改为"复杂度判断"
                "roles": [op.role.value for op in session.round1_opinions],
                "reason": (
                    f"复杂度: {session.complexity.level.value} | "
                    f"{session.complexity.reason}"
                ),
            }

        # 发送 Round 1 各角色独立意见
        for opinion in session.round1_opinions:
            yield {
                "type": "role_opinion",
                "role": opinion.role.value,
                "role_label": opinion.role_label,
                "content": opinion.content,
                "round": 1,
                "evidence": opinion.evidence,
            }

        # 发送 Round 2 反驳与修正意见
        for opinion in session.round2_opinions:
            yield {
                "type": "role_opinion",
                "role": opinion.role.value,
                "role_label": f"{opinion.role_label}（Round 2）",
                "content": opinion.content,
                "round": 2,
                "evidence": opinion.evidence,
            }

        # 协作有效性指标（论文核心）
        if session.round2_opinions:
            yield {
                "type": "collaboration_metric",
                "revision_rate": session.revision_rate,
                "disagreement_count": session.disagreement_count,
            }

        # 流式发送最终决策（Round 3 仲裁结果）
        synthesis = session.final_decision
        chunk_size = 40
        for i in range(0, len(synthesis), chunk_size):
            chunk = synthesis[i:i + chunk_size]
            yield {"type": "synthesis_token", "content": chunk}

        # 实验日志
        session_id = context.thread_id if context else "default"
        log_entry = ExperimentLog(
            session_id=session_id,
            experiment_mode=experiment_mode.value,
            query=message,
            roles_called=[op.role.value for op in session.round1_opinions],
            routing_reason=(
                session.complexity.reason if session.complexity else ""
            ),
            role_opinions={
                op.role.value: op.content for op in session.round1_opinions
            },
            final_answer=synthesis,
            guardian_verdict="safe" if experiment_mode.use_guardian else "skipped",
            latency_ms=session.latency_ms,
        )
        experiment_logger.log(log_entry)

        yield {"type": "done", "content": synthesis}

    async def generate_title(self, first_user_message: str) -> str:
        prompt = (
            "请根据用户的第一条消息生成一个中文会话标题。"
            "要求不超过 10 个汉字，不要带引号，不要解释。"
        )
        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": first_user_message},
                ]
            )
            title = _stringify_content(getattr(response, "content", "")).strip()
            return title[:10] or "新会话"
        except Exception:
            return (first_user_message.strip() or "新会话")[:10]

    async def summarize_history(self, messages: list[dict[str, Any]]) -> str:
        prompt = (
            "请将以下对话压缩成中文摘要，控制在 500 字以内。"
            "重点保留用户目标、已完成步骤、重要结论和未解决事项。"
        )
        lines: list[str] = []
        for item in messages:
            role = item.get("role", "assistant")
            content = str(item.get("content", "") or "")
            if content:
                lines.append(f"{role}: {content}")
        transcript = "\n".join(lines)

        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ]
            )
            summary = _stringify_content(getattr(response, "content", "")).strip()
            return summary[:500]
        except Exception:
            return transcript[:500]


agent_manager = AgentManager()
