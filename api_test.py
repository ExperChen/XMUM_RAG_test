# -*- coding: utf-8 -*-
"""
简单的 OpenRouter API 连通性测试脚本
用于快速检查 API 是否可以正常访问
"""

import requests
import json
from config import OPENROUTER_API_KEY

def test_api_connection():
    """
    测试 OpenRouter API 连通性
    发送一个简单的请求来检查 API 是否正常工作
    """
    
    print("正在测试 OpenRouter API 连通性...")
    print("-" * 40)
    
    # API 基本信息
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    
    # 简单的测试消息
    test_data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {
                "role": "user",
                "content": "Hello, just testing API connection. Please reply with 'API is working!'"
            }
        ],
        "max_tokens": 50  # 限制回复长度，节省 token
    }
    
    try:
        # 发送请求
        print(f"📡 发送请求到: {url}")
        print(f"🤖 使用模型: {test_data['model']}")
        print(f"💬 测试消息: {test_data['messages'][0]['content']}")
        print("\n⏳ 等待响应...")
        
        response = requests.post(
            url=url,
            headers=headers,
            json=test_data,
            timeout=15  # 15秒超时
        )
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            
            # 提取回答
            if 'choices' in result and len(result['choices']) > 0:
                api_response = result['choices'][0]['message']['content']
                
                print("\n✅ API 连通性测试成功！")
                print(f"📝 API 回复: {api_response}")
                
                # 显示使用统计
                if 'usage' in result:
                    usage = result['usage']
                    print(f"\n📊 使用统计:")
                    print(f"   输入 tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(f"   输出 tokens: {usage.get('completion_tokens', 'N/A')}")
                    print(f"   总计 tokens: {usage.get('total_tokens', 'N/A')}")
                
                return True
            else:
                print("❌ 错误：响应格式异常")
                print(f"完整响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return False
                
        else:
            print(f"❌ API 请求失败！")
            print(f"状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时！请检查网络连接")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请检查网络连接或 API 地址")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {str(e)}")
        return False
    except json.JSONDecodeError:
        print("❌ 响应不是有效的 JSON 格式")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
        return False

def ask_llm_question(question):
    """
    向 LLM 提问并获取回答
    参数:
        question (str): 要问的问题
    返回:
        bool: 是否成功获取回答
    """
    
    print(f"\n🤔 向 LLM 提问: {question}")
    print("-" * 50)
    
    # API 基本信息
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    
    # 构建问题数据
    question_data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "max_tokens": 200  # 允许更长的回答
    }
    
    try:
        print("⏳ 正在思考中...")
        
        response = requests.post(
            url=url,
            headers=headers,
            json=question_data,
            timeout=20  # 20秒超时，因为问题可能需要更多时间思考
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                llm_answer = result['choices'][0]['message']['content']
                
                print("\n🤖 LLM 回答:")
                print("=" * 50)
                print(llm_answer)
                print("=" * 50)
                
                # 显示使用统计
                if 'usage' in result:
                    usage = result['usage']
                    print(f"\n📊 本次问答使用统计:")
                    print(f"   输入 tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(f"   输出 tokens: {usage.get('completion_tokens', 'N/A')}")
                    print(f"   总计 tokens: {usage.get('total_tokens', 'N/A')}")
                
                return True
            else:
                print("❌ 错误：无法获取 LLM 回答")
                return False
        else:
            print(f"❌ 请求失败！状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 提问过程中发生错误: {str(e)}")
        return False

def quick_test():
    """
    快速测试函数
    只显示最基本的连通性结果
    """
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }
        data = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            print("✅ API 连通正常")
            return True
        else:
            print(f"❌ API 连通失败 (状态码: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ API 连通失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== OpenRouter API 连通性测试 ===\n")
    
    # 首先进行连通性测试
    print("开始详细测试...")
    connection_success = test_api_connection()
    
    # 如果连通性测试成功，则向 LLM 提问
    if connection_success:
        print("\n" + "=" * 60)
        print("🎉 API 连通性测试通过！现在向 LLM 提问...")
        print("=" * 60)
        
        # 预设的问题
        test_question = "请用简单的话解释一下什么是人工智能？"
        
        # 向 LLM 提问
        ask_success = ask_llm_question(test_question)
        
        if ask_success:
            print("\n✅ 问答测试完成！LLM 工作正常。")
        else:
            print("\n❌ 问答测试失败，但 API 连通性正常。")
    else:
        print("\n❌ API 连通性测试失败，无法进行问答测试。")
    
    print("\n测试完成！")