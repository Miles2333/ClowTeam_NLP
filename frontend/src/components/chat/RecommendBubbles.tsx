"use client";

import { Sparkles } from "lucide-react";

import { useAppStore } from "@/lib/store";

export function RecommendBubbles() {
  const { recommendations, sendMessage, isStreaming } = useAppStore();

  if (!recommendations.length) {
    return null;
  }

  return (
    <div className="panel rounded-xl p-4 mb-3">
      <div className="flex items-center gap-2 text-sm text-ink-soft mb-3">
        <Sparkles size={14} className="text-ocean" />
        <span>你可能想问</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {recommendations.map((rec) => (
          <button
            key={rec.id}
            type="button"
            disabled={isStreaming}
            onClick={() => void sendMessage(rec.text)}
            className="px-3 py-1.5 text-xs bg-frost hover:bg-sand border border-line rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-ink"
          >
            {rec.text}
          </button>
        ))}
      </div>
    </div>
  );
}
