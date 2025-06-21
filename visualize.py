import json
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rcParams

# 设置中文字体和负号显示
rcParams['font.family'] = 'WenQuanYi Zen Hei'
rcParams['axes.unicode_minus'] = False

MODELS = ["Qwen", "ChatGLM3"]
ROUNDS = 10
INPUT_DIR = "output"
OUTPUT_DIR = "visual"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取所有问题ID
def get_all_question_ids():
    for i in range(1, ROUNDS + 1):
        file = os.path.join(INPUT_DIR, f"{MODELS[0].lower()}_output_{i}.json")
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                return list(json.load(f).keys())
    return []

# 每题每轮响应时间数据
def load_response_times_per_question(rounds):
    data = {model: {} for model in MODELS}
    question_ids = get_all_question_ids()

    for i in range(1, rounds + 1):
        for model in MODELS:
            file = os.path.join(INPUT_DIR, f"{model.lower()}_output_{i}.json")
            if not os.path.exists(file):
                continue
            with open(file, "r", encoding="utf-8") as f:
                model_data = json.load(f)
                for qid in question_ids:
                    time_val = model_data.get(qid, {}).get("time")
                    if time_val is not None:
                        data[model].setdefault(qid, []).append(time_val)
    return data

# 每题每轮评分数据
def load_scores_per_question(rounds):
    data = {model: {} for model in MODELS}
    for i in range(1, rounds + 1):
        file = os.path.join(INPUT_DIR, f"judgment_output_{i}.json")
        if not os.path.exists(file):
            continue
        with open(file, "r", encoding="utf-8") as f:
            jd = json.load(f)
            for qid in jd:
                for model in MODELS:
                    score = jd[qid].get("score", {}).get(model)
                    if score is not None:
                        data[model].setdefault(qid, []).append(score)
    return data

# 每轮胜出模型统计
def load_winrate(rounds):
    win_counts = {model: 0 for model in MODELS}
    for i in range(1, rounds + 1):
        file = os.path.join(INPUT_DIR, f"judgment_output_{i}.json")
        if not os.path.exists(file):
            continue
        with open(file, "r", encoding="utf-8") as f:
            jd = json.load(f)
            for qid in jd:
                winner = jd[qid].get("preferred_model")
                if winner in win_counts:
                    win_counts[winner] += 1
    return win_counts

# 折线图：平均值
def plot_mean_line_per_question(data, title, ylabel, filename):
    question_ids = sorted(set().union(*[data[m].keys() for m in MODELS]))
    x = np.arange(len(question_ids))

    plt.figure(figsize=(10, 6))
    for model in MODELS:
        means = [np.mean(data[model].get(qid, [0])) for qid in question_ids]
        plt.plot(x, means, marker='o', label=model)

    plt.xticks(x, question_ids, rotation=45)
    plt.xlabel("问题编号")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.show()

# 折线图：方差
def plot_variance_line_per_question(data, title, ylabel, filename):
    question_ids = sorted(set().union(*[data[m].keys() for m in MODELS]))
    x = np.arange(len(question_ids))

    plt.figure(figsize=(10, 6))
    for model in MODELS:
        variances = [np.var(data[model].get(qid, [0])) for qid in question_ids]
        plt.plot(x, variances, marker='s', label=model)

    plt.xticks(x, question_ids, rotation=45)
    plt.xlabel("问题编号")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.show()

# 饼图：平均胜率
def plot_winrate_pie(win_counts, filename="winrate_pie.png"):
    labels = list(win_counts.keys())
    sizes = list(win_counts.values())

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("模型胜率占比")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.show()

# 主函数
def launch():
    print("🔍 正在加载数据...")
    response_data = load_response_times_per_question(ROUNDS)
    score_data = load_scores_per_question(ROUNDS)
    winrate = load_winrate(ROUNDS)

    print("📊 正在绘图...")
    plot_mean_line_per_question(response_data, "平均响应时间 - 题目", "平均响应时间（秒）", "avg_response_time_per_question.png")
    plot_variance_line_per_question(response_data, "响应时间方差 - 题目", "响应时间方差", "var_response_time_per_question.png")
    plot_mean_line_per_question(score_data, "平均评分 - 题目", "平均评分（1-10）", "avg_score_per_question.png")
    plot_variance_line_per_question(score_data, "评分方差 - 题目", "评分方差", "var_score_per_question.png")
    plot_winrate_pie(winrate, filename="winrate_pie.png")

    print("✅ 可视化已完成！图表保存在 /visual 目录。")

if __name__ == '__main__':
    print("🔍 正在加载数据...")
    response_data = load_response_times_per_question(ROUNDS)
    score_data = load_scores_per_question(ROUNDS)
    winrate = load_winrate(ROUNDS)

    print("📊 正在绘图...")
    plot_mean_line_per_question(response_data, "平均响应时间 - 题目", "平均响应时间（秒）", "avg_response_time_per_question.png")
    plot_variance_line_per_question(response_data, "响应时间方差 - 题目", "响应时间方差", "var_response_time_per_question.png")
    plot_mean_line_per_question(score_data, "平均评分 - 题目", "平均评分（1-10）", "avg_score_per_question.png")
    plot_variance_line_per_question(score_data, "评分方差 - 题目", "评分方差", "var_score_per_question.png")
    plot_winrate_pie(winrate, filename="winrate_pie.png")

    print("✅ 可视化已完成！图表保存在 /visual 目录。")
