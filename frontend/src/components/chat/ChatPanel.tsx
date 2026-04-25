"use client";

import { useEffect, useRef } from "react";

import { ChatInput } from "@/components/chat/ChatInput";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { RecommendBubbles } from "@/components/chat/RecommendBubbles";
import { useAppStore } from "@/lib/store";

export function ChatPanel() {
  const { messages, sendMessage, isStreaming, tokenStats, experimentMode } = useAppStore();
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const modeLabel: Record<string, string> = {
    single: "单 Agent 基线",
    multi_no_memory: "多 Agent (无共享记忆)",
    multi_memory: "多 Agent + 共享记忆",
    multi_full: "多 Agent + 记忆 + 守卫"
  };

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col gap-4">
      <div className="panel flex items-center justify-between rounded-[30px] px-5 py-4">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-[var(--color-ink-soft)]">
            ClawTeam 协作诊疗
          </p>
          <h2 className="text-lg font-semibold tracking-[-0.04em]">
            当前模式: {modeLabel[experimentMode] ?? experimentMode}
          </h2>
        </div>
        <div className="mono text-sm text-[var(--color-ink-soft)]">
          {tokenStats ? `${tokenStats.total_tokens} tokens` : "No metrics yet"}
        </div>
      </div>

      <div className="panel flex min-h-0 flex-1 flex-col rounded-[32px] p-5">
        <div className="flex-1 space-y-4 overflow-y-auto pr-2">
          {!messages.length && (
            <div className="rounded-[28px] border border-dashed border-[var(--color-line)] bg-white/45 p-8">
              <p className="text-xs uppercase tracking-[0.28em] text-[var(--color-ink-soft)]">
                ClawTeam Ready
              </p>
              <h3 className="mt-2 text-3xl font-semibold tracking-[-0.05em]">
                医疗多智能体协作诊疗系统
              </h3>
              <p className="mt-3 max-w-2xl text-[var(--color-ink-soft)]">
                主治医生 · 临床药师 · 影像科医生 协同会诊。共享长期记忆、安全守卫、可解释证据链。
                请描述你的症状或疑问，或点击下方推荐问题快速开始。
              </p>
              <div className="mt-5">
                <RecommendBubbles />
              </div>
            </div>
          )}

          {messages.map((message) => (
            <ChatMessage
              content={message.content}
              key={message.id}
              retrievals={message.retrievals}
              role={message.role}
              toolCalls={message.toolCalls}
              roleOpinions={message.roleOpinions}
              routing={message.routing}
              guardianBlocked={message.guardianBlocked}
            />
          ))}
          <div ref={endRef} />
        </div>
      </div>

      <ChatInput disabled={isStreaming} onSend={sendMessage} />
    </section>
  );
}
