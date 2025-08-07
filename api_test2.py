import requests
import json

# 发送请求前的提示
print("正在发送API请求...")

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "deepseek/deepseek-chat-v3-0324:free",
    "messages": [
      {
        "role": "user",
        "content": "你是谁？"
      }
    ],
    
  })
)

# 打印响应状态码
print(f"Status Code: {response.status_code}")

# 检查请求是否成功
if response.status_code == 200:
    print("请求成功！")
    
    # 解析并打印JSON响应
    response_json = response.json()
    print("\nResponse:")
    print(json.dumps(response_json, indent=2, ensure_ascii=False))
    
    # 提取并打印AI的回答
    try:
        ai_response = response_json['choices'][0]['message']['content']
        print("\nAI's Answer:")
        print(ai_response)
    except KeyError:
        print("\n无法从响应中提取AI的回答")
else:
    print("\n请求失败")
    print(f"错误信息: {response.text}")  # 添加错误信息打印

print("\n程序执行完成")  # 添加程序结束提示

