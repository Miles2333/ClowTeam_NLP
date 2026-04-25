"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Stethoscope, Pill, Scan } from "lucide-react";

import type { RoleOpinion } from "@/lib/api";

const ROLE_META: Record<string, { icon: JSX.Element; color: string; border: string }> = {
  physician: {
    icon: <Stethoscope size={16} />,
    color: "text-ocean",
    border: "border-ocean/40"
  },
  pharmacist: {
    icon: <Pill size={16} />,
    color: "text-ember",
    border: "border-ember/40"
  },
  radiologist: {
    icon: <Scan size={16} />,
    color: "text-ink",
    border: "border-ink/30"
  }
};

export function RoleOpinionCard({ opinion }: { opinion: RoleOpinion }) {
  const meta = ROLE_META[opinion.role] ?? {
    icon: <Stethoscope size={16} />,
    color: "text-ink",
    border: "border-line"
  };

  return (
    <details
      className={`panel rounded-xl border ${meta.border} mb-2 overflow-hidden`}
      open
    >
      <summary
        className={`flex items-center gap-2 px-4 py-2 cursor-pointer select-none ${meta.color} font-medium`}
      >
        {meta.icon}
        <span>{opinion.role_label}</span>
        <span className="text-ink-soft text-xs ml-auto">角色会诊意见</span>
      </summary>
      <div className="markdown px-4 py-3 text-sm border-t border-line">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{opinion.content}</ReactMarkdown>
        {opinion.evidence && opinion.evidence.length > 0 && (
          <div className="mt-2 pt-2 border-t border-line/50 text-xs text-ink-soft">
            <div className="font-medium mb-1">证据来源：</div>
            <ul className="list-disc list-inside">
              {opinion.evidence.map((src, idx) => (
                <li key={idx}>{src}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </details>
  );
}
