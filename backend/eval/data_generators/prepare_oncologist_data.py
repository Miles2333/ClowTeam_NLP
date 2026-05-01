"""准备肿瘤内科训练数据：真实 Benchmark 筛选 + LLM 改写扩充。

与 surgeon 同结构，但筛选关键词换成内科相关（化疗/靶向/免疫等）。
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
from eval.datasets._oncology_keywords import is_oncologist_topic

ONCOLOGY_DIR = DATA_DIR / "oncology"
TRAINING_DIR = DATA_DIR / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = TRAINING_DIR / "oncologist_train.jsonl"
TARGET_AUGMENT_COUNT = 400
ROLE_PROMPT_PATH = BACKEND_DIR / "workspace" / "roles" / "MEDICAL_ONCOLOGIST.md"


def load_role_prompt() -> str:
    if ROLE_PROMPT_PATH.exists():
        return ROLE_PROMPT_PATH.read_text(encoding="utf-8")
    return "你是肿瘤内科医生，参与 Tumor Board 会诊。"


def collect_real_data() -> list[dict]:
    real = []

    # MedQA
    for split in ["train", "validation", "test"]:
        path = ONCOLOGY_DIR / f"medqa_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = (item.get("question", "")
                    + " " + " ".join(str(o) for o in item.get("options", []))
                    + " " + str(item.get("answer_text", "")))
            if is_oncologist_topic(text):
                real.append({
                    "source": "MedQA",
                    "question": item["question"],
                    "options": item.get("options", []),
                    "answer": item.get("answer_text", ""),
                    "raw": item,
                })

    # CMExam
    for split in ["train", "validation", "test"]:
        path = ONCOLOGY_DIR / f"cmexam_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            text = json.dumps(item, ensure_ascii=False)
            if is_oncologist_topic(text):
                real.append({
                    "source": "CMExam",
                    "question": item.get("Question", item.get("question", "")),
                    "options": item.get("Options", item.get("options", [])),
                    "answer": item.get("Answer", item.get("answer", "")),
                    "explanation": item.get("Explanation", ""),
                    "raw": item,
                })

    # PubMedQA 肿瘤子集（QA 形式，作为补充）
    for split in ["train", "validation", "test"]:
        path = ONCOLOGY_DIR / f"pubmedqa_{split}_oncology.jsonl"
        if not path.exists():
            continue
        for item in load_jsonl(path):
            real.append({
                "source": "PubMedQA",
                "question": item.get("question", ""),
                "answer": item.get("long_answer", item.get("final_decision", "")),
                "raw": item,
            })

    print(f"[oncologist] 真实数据：{len(real)} 条")
    return real


def to_chatml(record: dict, role_prompt: str) -> dict | None:
    question = record.get("question", "").strip()
    if not question:
        return None

    options = record.get("options", [])
    answer = record.get("answer", "").strip()
    explanation = record.get("explanation", "").strip()

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


def build_augment_prompt(seed_records: list[dict]) -> tuple[str, str]:
    seeds_text = "\n\n".join(
        f"原题：{r['question']}\n标准答案：{r.get('answer', '')}"
        for r in seed_records
    )

    system = (
        "你是医学训练数据生成员。把医学考试题改写成 Tumor Board 肿瘤内科医生回答风格。"
        "回答按内科医生结构化输出（治疗目标 / 推荐方案 / 剂量周期 / 不良反应 / 与其他治疗协调）。"
        "输出 JSON。"
    )

    user = f"""请基于下面 3 道真实考题，改写成 3 对肿瘤内科医生训练对话（保留正确答案的医学事实）：

{seeds_text}

要求：
1. 把原题改写成口语化的 MDT 病例咨询
2. 答案按内科医生结构化输出（治疗目标→推荐方案→剂量周期→不良反应→与其他治疗协调）
3. 严格基于原题的医学事实，不要编造剂量
4. 长度 200-400 字

输出 JSON：
{{
  "items": [{{"user": "...", "assistant": "..."}}, ...]
}}
"""
    return system, user


def augment_with_llm(real: list[dict], role_prompt: str, target: int) -> int:
    if len(real) < 3:
        return 0

    existing = load_jsonl(OUTPUT)
    augmented_count = sum(1 for r in existing if r.get("source") == "augmented")
    seen_users = {r["messages"][1]["content"] for r in existing if "messages" in r}

    while augmented_count < target:
        seed = random.sample(real, 3)
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

            print(f"[oncologist-aug] 进度 {augmented_count}/{target}")
        except Exception as exc:
            print(f"[oncologist-aug] {exc}")
            continue

    return augmented_count


def main():
    role_prompt = load_role_prompt()

    if not OUTPUT.exists() or not load_jsonl(OUTPUT):
        real = collect_real_data()
        if not real:
            print("[oncologist] ⚠️ 真实数据为空，请先下载 benchmark")
            return
        for r in real:
            chatml = to_chatml(r, role_prompt)
            if chatml is not None:
                append_jsonl(chatml, OUTPUT)
        print(f"[oncologist] 真实数据 ChatML 写入 {OUTPUT}")

    real = collect_real_data()
    augment_with_llm(real, role_prompt, TARGET_AUGMENT_COUNT)

    final = load_jsonl(OUTPUT)
    print(f"\n[oncologist] ✅ 共 {len(final)} 条")
    print(f"  真实: {sum(1 for r in final if r.get('source') != 'augmented')}")
    print(f"  扩充: {sum(1 for r in final if r.get('source') == 'augmented')}")


if __name__ == "__main__":
    random.seed(42)
    main()
