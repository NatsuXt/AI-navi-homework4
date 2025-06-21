# score_judge.py
import json
import re
import os
from llm_openai import stream_llm_response

OUTPUT_DIR = "output"

def score_judgment_for_pair(qwen_file, glm_file, output_file, question_file="questions.json", model="gpt-4o"):

    # 加载数据
    with open(qwen_file, "r", encoding="utf-8") as f:
        qwen_data = json.load(f)
    with open(glm_file, "r", encoding="utf-8") as f:
        glm_data = json.load(f)
    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = {}

    for qid in questions:
        question = questions[qid]
        answer_qwen = qwen_data.get(qid, {}).get("answer", "")
        answer_glm = glm_data.get(qid, {}).get("answer", "")

        system_prompt = "你是一位语言理解专家，需要对两个模型的回答进行打分和分析。"
        user_prompt = f"""
请你根据以下问题和两个模型的回答，从准确性、逻辑性、表达清晰度等方面进行比较评分。

问题：{question}

模型A（Qwen）的回答：
{answer_qwen}

模型B（ChatGLM3）的回答：
{answer_glm}

请你输出：
更优模型：A/B
评分：A: x, B: y（x 和 y 为 1 到 10 的整数）
评语：简要说明理由
"""

        print(f"\n🔍 正在评估 {qid}...\n")
        reply = ""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            for chunk in stream_llm_response(messages, model=model):
                print(chunk, end="", flush=True)
                reply += chunk
            print("\n")

            # 解析回复内容
            preferred = "Qwen" if re.search(r"更优模型[:：]?\s*A", reply) else "ChatGLM3"
            score_match = re.search(r"A[:：]?\s*(\d+)[^0-9]+B[:：]?\s*(\d+)", reply)
            score_a = int(score_match.group(1)) if score_match else None
            score_b = int(score_match.group(2)) if score_match else None
            comment_start = reply.find("评语：")
            comment = reply[comment_start + 3:].strip() if comment_start != -1 else "无"

            results[qid] = {
                "preferred_model": preferred,
                "score": {
                    "Qwen": score_a,
                    "ChatGLM3": score_b
                },
                "comment": comment
            }

        except Exception as e:
            results[qid] = {
                "error": str(e)
            }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 评分完成：{output_file}")
