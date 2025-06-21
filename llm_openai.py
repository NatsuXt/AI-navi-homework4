# modules/llm_openai.py
import json
import openai
from openai import OpenAI
from pathlib import Path

config = json.loads(Path("api_config.json").read_text(encoding="utf-8"))
api_key = config["OPENAI_API_KEY"]
api_base = config.get("API_BASE_URL", "https://api.openai.com/v1")
client = OpenAI(api_key=api_key, base_url=api_base)

def stream_llm_response(messages, model="gpt-4o"):
    """
    使用 OpenAI >= 1.x API 流式生成内容
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        content = delta.content if delta else ""
        if content:
            yield content
