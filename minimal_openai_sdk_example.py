from openai import OpenAI

client = OpenAI(
    api_key="sk-xxxx",
    base_url="https://api.ikuncode.cc/v1",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"},
    ],
    temperature=0.7,
    max_tokens=1024,
)

print(response.choices[0].message.content)
