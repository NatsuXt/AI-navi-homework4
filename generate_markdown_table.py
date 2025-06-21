import json
import os
from collections import defaultdict, Counter
import numpy as np

def generate(rounds=10, models=["Qwen", "ChatGLM3"], input_dir="output", output_dir="visual", output_filename="model_score_summary.md"):
    """
    汇总多个轮次评分并生成 Markdown 表格报告

    参数:
        rounds (int): 总轮次数
        models (list): 模型名称列表
        input_dir (str): 输入 JSON 文件目录
        output_dir (str): 输出目录
        output_filename (str): 输出 Markdown 文件名
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # 初始化数据结构
    scores = {model: defaultdict(list) for model in models}
    wins = defaultdict(list)
    comments = defaultdict(list)

    # 读取所有轮次评分
    for i in range(1, rounds + 1):
        path = os.path.join(input_dir, f"judgment_output_{i}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ JSON解析失败: {path}")
                continue
            for qid, entry in data.items():
                for model in models:
                    score = entry.get("score", {}).get(model)
                    if score is not None:
                        scores[model][qid].append(score)
                winner = entry.get("preferred_model")
                if winner:
                    wins[qid].append(winner)
                if "comment" in entry:
                    clean_comment = entry["comment"].replace("|", "\\|").replace("\n", " ").strip()
                    comments[qid].append(clean_comment)

    # 构造Markdown表格
    lines = []
    header = "| 问题编号 | " + " | ".join([f"{m} 平均分" for m in models]) + " | " \
             + " | ".join([f"{m}胜" for m in models]) + " | 最常胜者 | 示例评语 |"
    sep = "|" + "----------|" * (len(models) * 2 + 2) + "-----------|"
    lines.append(header)
    lines.append(sep)

    all_qids = sorted(set().union(*[scores[m].keys() for m in models]))
    for qid in all_qids:
        row = [qid]
        # 平均分
        for model in models:
            model_scores = scores[model].get(qid, [])
            avg = np.mean(model_scores) if model_scores else 0.0
            row.append(f"{avg:.2f}")
        # 胜场
        win_list = wins.get(qid, [])
        win_counts = Counter(win_list)
        for model in models:
            row.append(str(win_counts.get(model, 0)))
        # 最常胜者
        if len(models) >= 2:
            max_count = max(win_counts.get(m, 0) for m in models)
            top_models = [m for m in models if win_counts.get(m, 0) == max_count]
            dominant = top_models[0] if len(top_models) == 1 else "平局"
        else:
            dominant = models[0]
        row.append(dominant)
        # 示例评语
        example_comment = comments[qid][0] if comments[qid] else "无"
        if len(example_comment) > 40:
            example_comment = example_comment[:37] + "..."
        row.append(example_comment)

        lines.append("| " + " | ".join(row) + " |")

    # 写入Markdown文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 已导出评分汇总至 Markdown：{output_path}")