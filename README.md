
# AI 模型评估系统

该项目用于自动化比较两个大语言模型（如 Qwen 和 ChatGLM3）的输出，执行推理、评分、可视化和 Markdown 报告生成。

## 📦 环境配置

1. 创建并激活 Conda 虚拟环境：

```bash
conda create -n ainavi_env python=3.10 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate ainavi_env
```

2. 安装 PyTorch（请根据你的环境下载相应的 `.whl` 文件）：

```bash
pip install torch-2.3.0+cu121-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.18.0+cu121-cp310-cp310-linux_x86_64.whl
```

3. 安装基础依赖：

在安装依赖前，建议检查 pip 能否联网：

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

4. 安装 `fschat`（需要启用 PEP517 构建）：

```bash
pip install fschat --use-pep517
```

## 🛠️ 项目结构说明

- `requirements.txt` : 环境需求，兼容了新旧transformer在pad参数上的不一致问题
- `main.py`：主程序，执行模型推理、评分和可视化。
- `score_judge.py`：使用 GPT 模型对两个模型回答进行打分和优劣评判。
- `visualize.py`：生成响应时间、评分方差等可视化图表。
- `generate_markdown_table.py`：生成模型评分汇总的 Markdown 表格报告。
- `config.json`：定义两个模型的路径和加载方式。
- `api_config.json`：定义GPT模型接口数据（**请在本地测试时替换为自己的api，如有需要请联系我**）。
- `questions.json`：需要进行测试的问题列表。

## 🚀 运行方式

确保你已准备好 `questions.json` 、 `api_config.json` 和 `config.json` 文件后，在环境中执行：

```bash
python main.py
```

系统将：
- 加载两个模型；
- 对每个问题执行 10 轮推理；
- 自动对比并评分；
- 输出可视化图表；
- 生成 Markdown 报告文件 `visual/model_score_summary.md`。

## 📁 输出说明

- 推理结果保存于：`output/` 目录
- 可视化图表保存在：`visual/` 目录
- Markdown 汇总表格为：`visual/model_score_summary.md`

---

如需进一步定制模型、评分逻辑或可视化方式，请参考各模块源码进行修改。需要我生成 `requirements.txt` 吗？
