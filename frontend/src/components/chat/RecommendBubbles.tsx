"use client";

import { Sparkles, Stethoscope, ChevronRight } from "lucide-react";

import { useAppStore } from "@/lib/store";
import type { Recommendation } from "@/lib/api";

// 不同肿瘤类型的颜色映射（让卡片视觉有差异）
const TUMOR_COLORS: Record<string, string> = {
  lung: "bg-ocean/10 text-ocean border-ocean/30",
  gastric: "bg-ember/10 text-ember border-ember/30",
  liver: "bg-amber-100 text-amber-800 border-amber-300",
  breast: "bg-pink-100 text-pink-800 border-pink-300",
  colorectal: "bg-emerald-100 text-emerald-800 border-emerald-300",
  esophageal: "bg-purple-100 text-purple-800 border-purple-300",
  pancreatic: "bg-indigo-100 text-indigo-800 border-indigo-300",
  thyroid: "bg-cyan-100 text-cyan-800 border-cyan-300",
  prostate: "bg-blue-100 text-blue-800 border-blue-300",
  ovarian: "bg-rose-100 text-rose-800 border-rose-300",
  cervical: "bg-fuchsia-100 text-fuchsia-800 border-fuchsia-300",
  lymphoma: "bg-violet-100 text-violet-800 border-violet-300",
  head_neck: "bg-orange-100 text-orange-800 border-orange-300",
  renal: "bg-teal-100 text-teal-800 border-teal-300",
  sarcoma: "bg-slate-100 text-slate-800 border-slate-300",
};

function tumorTypeColor(type?: string): string {
  if (!type) return "bg-frost text-ink border-line";
  return TUMOR_COLORS[type] || "bg-frost text-ink border-line";
}

export function RecommendBubbles() {
  const { recommendations, sendMessage, isStreaming } = useAppStore();

  if (!recommendations.length) {
    return null;
  }

  function handleClick(rec: Recommendation) {
    if (isStreaming) return;
    // 点击：把完整病例作为用户消息发送
    const message = rec.case_data?.case || rec.text;
    void sendMessage(message);
  }

  return (
    <div className="panel rounded-xl p-4 mb-3">
      <div className="flex items-center gap-2 text-sm text-ink-soft mb-3">
        <Sparkles size={14} className="text-ocean" />
        <span className="font-medium">推荐 Tumor Board 案例</span>
        <span className="text-xs text-ink-soft/70 ml-auto">
          基于历史 + 共享记忆 + 角色权重
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {recommendations.map((rec) => {
          const tumorType = rec.case_data?.tumor_type;
          const colorClass = tumorTypeColor(tumorType);
          const stage = rec.case_data?.stage;
          const biomarker = rec.case_data?.biomarker;

          return (
            <button
              key={rec.id}
              type="button"
              disabled={isStreaming}
              onClick={() => handleClick(rec)}
              className={`group flex flex-col gap-1.5 px-3 py-2.5 border rounded-lg text-left transition-all hover:shadow-sm hover:scale-[1.01] disabled:opacity-50 disabled:cursor-not-allowed ${colorClass}`}
            >
              <div className="flex items-start gap-2">
                <Stethoscope size={14} className="mt-0.5 shrink-0" />
                <span className="text-sm font-medium leading-snug flex-1">
                  {rec.text}
                </span>
                <ChevronRight
                  size={14}
                  className="mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                />
              </div>
              {(stage || biomarker) && (
                <div className="flex items-center gap-2 text-xs opacity-80 ml-5">
                  {stage && <span>{stage}</span>}
                  {biomarker && <span>· {biomarker}</span>}
                </div>
              )}
              {rec.reason && (
                <div className="text-xs opacity-70 ml-5 italic">
                  💡 {rec.reason}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
