"use client";

import { SendHorizonal } from "lucide-react";
import { useState } from "react";

export function ChatInput({
  disabled,
  onSend
}: {
  disabled: boolean;
  onSend: (value: string) => Promise<void>;
}) {
  const [value, setValue] = useState("");

  function submit() {
    const nextValue = value.trim();
    if (!nextValue) {
      return;
    }
    void onSend(nextValue);
    setValue("");
  }

  return (
    <div className="panel rounded-lg p-2">
      <div className="flex items-end gap-2">
        <textarea
          className="max-h-32 min-h-12 flex-1 resize-y rounded-lg border border-[var(--color-line)] bg-white/82 px-3 py-2 text-sm leading-6 outline-none focus:border-ocean/50 focus:bg-white"
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
              event.preventDefault();
              submit();
            }
          }}
          placeholder="输入病例、分期、病理、基因检测和需要讨论的问题。Ctrl / Cmd + Enter 发送。"
          rows={2}
          value={value}
        />
        <button
          className="flex h-12 shrink-0 items-center gap-2 rounded-lg bg-ocean px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-ocean/45"
          disabled={disabled || !value.trim()}
          onClick={submit}
          type="button"
        >
          <SendHorizonal size={16} />
          发送
        </button>
      </div>
      <p className="mt-1 px-1 text-[11px] leading-4 text-[var(--color-ink-soft)]">
        仅供科研和辅助讨论使用，具体诊疗请以临床医生判断为准。
      </p>
    </div>
  );
}
