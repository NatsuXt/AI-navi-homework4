import json
import torch
import time
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from score_judge import score_judgment_for_pair
import visualize
import generate_markdown_table

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型（仅加载一次）
def load_model_once(model_name, model_path, mode):
    print(f"[{model_name}] 加载模型中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if mode == "chatglm":
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device).eval()

    return tokenizer, model, device


# 执行一轮问题推理
def run_questions_batch(model_name, tokenizer, model, device, questions, output_path, mode):
    result = {}
    for qid, question in questions.items():
        try:
            start = time.time()
            if mode == "chatglm":
                response, _ = model.chat(tokenizer, question, history=[])
            else:
                inputs = tokenizer.encode(question, return_tensors="pt").to(device)
                outputs = model.generate(inputs, max_new_tokens=300)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            end = time.time()

            result[qid] = {
                "answer": response.strip(),
                "time": round(end - start, 2)
            }

        except Exception as e:
            result[qid] = {
                "answer": f"发生错误: {str(e)}",
                "time": None
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{model_name}] 已完成写入: {output_path}")


# 运行 10 轮测试，复用同一个模型
def run_ten_rounds(model_cfg, questions):
    model_name = model_cfg["model_name"]
    model_path = model_cfg["model_path"]
    mode = model_cfg.get("mode", "qwen")

    tokenizer, model, device = load_model_once(model_name, model_path, mode)

    for i in range(1, 11):
        output_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_output_{i}.json")
        run_questions_batch(model_name, tokenizer, model, device, questions, output_path, mode)


# 执行所有评分
def run_all_judgments():
    for i in range(1, 11):
        qwen_file = os.path.join(OUTPUT_DIR, f"qwen_output_{i}.json")
        glm_file = os.path.join(OUTPUT_DIR, f"chatglm3_output_{i}.json")
        output_file = os.path.join(OUTPUT_DIR, f"judgment_output_{i}.json")
        score_judgment_for_pair(qwen_file, glm_file, output_file)


if __name__ == "__main__":
    with open("questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)

    run_ten_rounds(configs[0], questions)
    torch.cuda.empty_cache()

    run_ten_rounds(configs[1], questions)

    run_all_judgments()

    print("\n✅ 所有模型推理与评分完成，结果保存在 /output 目录中")
    visualize.launch()
    generate_markdown_table.generate()
