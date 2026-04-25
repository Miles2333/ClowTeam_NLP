"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ShieldAlert, Users } from "lucide-react";

import { RetrievalCard } from "@/components/chat/RetrievalCard";
import { RoleOpinionCard } from "@/components/chat/RoleOpinionCard";
import { ThoughtChain } from "@/components/chat/ThoughtChain";
import type { RetrievalResult, RoleOpinion, RoutingInfo, ToolCall } from "@/lib/api";

export function ChatMessage({
  role,
  content,
  toolCalls,
  retrievals,
  roleOpinions,
  routing,
  guardianBlocked
}: {
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievals: RetrievalResult[];
  roleOpinions?: RoleOpinion[];
  routing?: RoutingInfo | null;
  guardianBlocked?: { reason: string; message: string } | null;
}) {
  const isUser = role === "user";
  const hasRoleOpinions = !!(roleOpinions && roleOpinions.length > 0);

  return (
    <article
      className={`max-w-[90%] rounded-[28px] px-5 py-4 ${
        isUser
          ? "ml-auto bg-[rgba(13,37,48,0.92)] text-white"
          : "panel mr-auto text-[var(--color-ink)]"
      }`}
    >
      {!isUser && guardianBlocked && (
        <div className="mb-3 flex items-start gap-2 rounded-xl border border-ember/40 bg-ember/10 p-3 text-sm">
          <ShieldAlert size={16} className="mt-0.5 text-ember" />
          <div>
            <div className="font-medium text-ember">安全守卫已拦截此请求</div>
            <div className="mt-1 text-xs text-ink-soft">原因: {guardianBlocked.reason}</div>
            <div className="mt-1 text-ink">{guardianBlocked.message}</div>
          </div>
        </div>
      )}

      {!isUser && <RetrievalCard results={retrievals} />}

      {!isUser && routing && routing.roles.length > 0 && (
        <div className="mb-2 flex items-center gap-2 rounded-full border border-ocean/30 bg-ocean/5 px-3 py-1 text-xs text-ocean w-fit">
          <Users size={12} />
          <span>本次会诊: {routing.roles.join(" + ")}</span>
        </div>
      )}

      {!isUser && hasRoleOpinions && (
        <div className="mb-3">
          {roleOpinions!.map((opinion, idx) => (
            <RoleOpinionCard key={`${opinion.role}-${idx}`} opinion={opinion} />
          ))}
        </div>
      )}

      {!isUser && <ThoughtChain toolCalls={toolCalls} />}

      {hasRoleOpinions && !isUser && content && content.trim() !== "" && (
        <div className="mb-2 text-xs uppercase tracking-[0.2em] text-ink-soft">
          综合结论
        </div>
      )}

      {(!isUser && toolCalls.length > 0 && (!content || content.trim() === "") && !toolCalls.some(tc => tc.output === content)) ? null : (
        (content && content.trim() !== "" && (!toolCalls.length || !toolCalls.some(tc => tc.output.trim() === content.trim()))) && (
          <div className={isUser ? "whitespace-pre-wrap leading-7" : "markdown"}>
            {isUser ? (
              content
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
            )}
          </div>
        )
      )}
      {!isUser && (!content || content.trim() === "") && !toolCalls.length && !hasRoleOpinions && !guardianBlocked && (
        <div className="text-[var(--color-ink-soft)]">正在思考...</div>
      )}
    </article>
  );
}
