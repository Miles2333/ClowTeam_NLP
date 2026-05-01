"""准备肿瘤外科训练数据：真实 Benchmark 筛选 + LLM 改写扩充。

来源：
- MedQA-USMLE 肿瘤外科题（~600 条）
- MedBullets 肿瘤外科病例（~200 条，可选）
- CMExam 中文肿瘤外科题（~400 条）
- DeepSeek 改写扩充（用真实题做 seed，~400 条）

输出格式：ChatML 格式（含 system/user/assistant），可直接用于 Qwen3 LoRA 训练。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _common import (
    BACKEND_DIR,
    DATA_DIR,
    append_jsonl,
    call_llm,
    load_jsonl,
    parse_json_safely,
)
from eval.datasets._oncology_keywords import is_surgeon_topic

ONCOLOGY_DIR = DATA_DIR / "oncology"
TRAINING_DIR = DATA_DIR / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = TRAINING_DIR / "surgeon_train.jsonl"
TARGET_AUGMENT_COUNT = 400  # LLM 扩充的目标
ROLE_PROMPT_PATH = BACKEND_DIR / "workspace" / "roles" / "SURGEON.md"


def load_role_prompt() -> str:
    if ROLE_PROMPT_PATH.exists():
        return ROLE_PROMPT_PATH.read_text(encoding="utf-8")
    return "你是肿瘤外科医生，参与 Tumor Board 会诊。"


# ───────────── Step 1: 从 Benchmark 筛选真实数据 ─────────────

def collect_real_data() -> list[dict]:
    """从已下载的肿瘤子集中筛选外科相关题。"""
    real_records = []

    # MedQA 肿瘤子集
    for split in ["train", "validation", "test"]:
        path = ONCOLOGY_DIR / f"medqa_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = (item.get("question", "")
                    + " " + " ".join(str(o) for o in item.get("options", []))
                    + " " + str(item.get("answer_text", "")))
            if is_surgeon_topic(text):
                real_records.append({
                    "source": "MedQA",
                    "question": item["question"],
                    "options": item.get("options", []),
                    "answer": item.get("answer_text", ""),
                    "raw": item,
                })

    # CMExam 肿瘤子集
    for split in ["train", "validation", "test"]:
        path = ONCOLOGY_DIR / f"cmexam_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = json.dumps(item, ensure_ascii=False)
            if is_surgeon_topic(text):
                real_records.append({
                    "source": "CMExam",
                    "question": item.get("Question", item.get("question", "")),
                    "options": item.get("Options", item.get("options", [])),
                    "answer": item.get("Answer", item.get("answer", "")),
                    "explanation": item.get("Explanation", ""),
                    "raw": item,
                })

    # MedBullets 肿瘤子集（如有）
    for split in ["train", "validation", "test"]:
        path = ONCOLOGY_DIR / f"medbullets_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = json.dumps(item, ensure_ascii=False)
            if is_surgeon_topic(text):
                real_records.append({
                    "source": "MedBullets",
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "raw": item,
                })

    print(f"[surgeon] 真实数据筛选：{len(real_records)} 条")
    return real_records


# ───────────── Step 2: 转换为 ChatML 格式 ─────────────

def to_chatml(record: dict, role_prompt: str) -> dict | None:
    """将真实题转为 ChatML 格式：system + user(病例/问题) + assistant(标准答+解析)"""
    question = (record.get("question") or "").strip()
    if not question:
        return None

    options = record.get("options") or []
    answer = (record.get("answer") or "").strip()
    explanation = (record.get("explanation") or "").strip()

    if isinstance(options, list) and options:
        opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
        user_text = f"{question}\n\n{opt_text}"
    else:
        user_text = question

    if explanation:
        assistant_text = f"答案：{answer}\n\n解析：{explanation}"
    else:
        assistant_text = f"答案：{answer}" if answer else ""

    if not assistant_text:
        return None

    return {
        "messages": [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "source": record["source"],
    }


# ───────────── Step 3: LLM 改写扩充（用真实题做 seed）─────────────

def build_augment_prompt(seed_records: list[dict]) -> tuple[str, str]:
    """让 DeepSeek 把真实题改写成 MDT 讨论格式。"""
    seeds_text = "\n\n".join(
        f"原题：{r['question']}\n标准答案：{r.get('answer', '')}"
        for r in seed_records
    )

    system = (
        "你是医学训练数据生成员。你需要把医学考试题改写成 Tumor Board "
        "（肿瘤多学科会诊）外科医生回答风格的训练样本。"
        "回答要遵循外科医生 prompt 中定义的结构（可切除性 / 术式 / 淋巴清扫 / 围手术期）。"
        "输出 JSON。"
    )

    user = f"""请基于下面 3 道真实考题，改写成 3 对外科医生训练对话（保留正确答案的医学事实）：

{seeds_text}

要求：
1. 把原题改写成口语化的 MDT 病例咨询（如"58 岁男性，肺腺癌 T2N1M0..."）
2. 答案按外科医生结构化输出（可切除性 → 术式 → 淋巴清扫 → 围手术期）
3. 严格基于原题的医学事实，不要编造
4. 长度 200-400 字

输出 JSON：
{{
  "items": [
    {{"user": "...", "assistant": "..."}},
    ...
  ]
}}
"""
    return system, user


def augment_with_llm(real_records: list[dict], role_prompt: str, target: int) -> int:
    """用真实题做 seed，让 LLM 改写扩充。"""
    if len(real_records) < 3:
        print("[surgeon] 真实数据不足 3 条，无法做 seed 扩充")
        return 0

    existing = load_jsonl(OUTPUT)
    augmented_count = sum(1 for r in existing if r.get("source") == "augmented")
    seen_users = {r["messages"][1]["content"] for r in existing if "messages" in r}

    while augmented_count < target:
        seed = random.sample(real_records, 3)
        system, user = build_augment_prompt(seed)

        try:
            response = call_llm(system, user, temperature=0.85, response_format_json=True)
            payload = parse_json_safely(response)
            if not payload or "items" not in payload:
                continue

            for item in payload["items"]:
                user_text = (item.get("user") or "").strip()
                asst_text = (item.get("assistant") or "").strip()
                if not user_text or not asst_text or user_text in seen_users:
                    continue

                record = {
                    "messages": [
                        {"role": "system", "content": role_prompt},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": asst_text},
                    ],
                    "source": "augmented",
                }
                append_jsonl(record, OUTPUT)
                seen_users.add(user_text)
                augmented_count += 1
                if augmented_count >= target:
                    break

            print(f"[surgeon-aug] 进度 {augmented_count}/{target}")
        except Exception as exc:
            print(f"[surgeon-aug] {exc}")
            continue

    return augmented_count


# ───────────── Main ─────────────

def main():
    role_prompt = load_role_prompt()

    # Step 1+2: 真实数据 → ChatML
    if not OUTPUT.exists() or not load_jsonl(OUTPUT):
        real = collect_real_data()
        if not real:
            print("[surgeon] ⚠️ 真实数据为空。请先跑：")
            print("  python eval/datasets/download_medqa.py")
            print("  python eval/datasets/download_cmexam.py")
            return

        for r in real:
            chatml = to_chatml(r, role_prompt)
            if chatml is not None:
                append_jsonl(chatml, OUTPUT)
        print(f"[surgeon] 真实数据 ChatML 已写入 {OUTPUT}")

    # Step 3: LLM 扩充
    real = collect_real_data()
    augment_with_llm(real, role_prompt, TARGET_AUGMENT_COUNT)

    final = load_jsonl(OUTPUT)
    print(f"\n[surgeon] ✅ 完成，共 {len(final)} 条训练数据")
    print(f"  真实: {sum(1 for r in final if r.get('source') != 'augmented')}")
    print(f"  扩充: {sum(1 for r in final if r.get('source') == 'augmented')}")


if __name__ == "__main__":
    random.seed(42)
    main()
