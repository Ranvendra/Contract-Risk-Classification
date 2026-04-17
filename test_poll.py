import json
from openai import OpenAI

client = OpenAI(base_url="https://text.pollinations.ai/openai", api_key="dummy")
res = client.chat.completions.create(
    model="openai",
    messages=[{"role": "user", "content": "return json { \"hello\": \"world\" }"}],
    response_format={"type": "json_object"}
)
print(res.choices[0].message.content)
