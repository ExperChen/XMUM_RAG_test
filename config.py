"""
RAG系统配置文件 - 使用OpenRouter服务
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 必须在import torch前
import torch
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import sys

# 设置默认编码为UTF-8
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    except:
        pass

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# OpenRouter配置
# API_KEY = "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" #Openrouter
# API_KEY = "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" #Openrouter
API_KEY = "sk-qtqjxthsswstjrgyeuymroiavwqzaddcoadsizjrhsoetidy" # Siliconflow
# BASE_URL = "https://openrouter.ai/api/v1"
BASE_URL = "https://api.siliconflow.cn/v1"
APP_NAME = "RAG Learning Project"  # 使用英文避免编码问题
APP_URL = "https://localhost:3000"

# 在文件顶部定义默认模型
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # 只需要在这里改

def get_openrouter_langchain_model(model=None):
    """
    获取用于LangChain的OpenRouter模型
    参数：
        model: 要使用的模型名称，如果为None则使用默认模型
    返回：配置好的ChatOpenAI模型
    """
    if model is None:
        model = DEFAULT_MODEL  # 使用默认模型
    
    return ChatOpenAI(
        model=model,
        
        api_key=API_KEY,
        
        base_url=BASE_URL,
        temperature=0.3,
        max_tokens=1500,
        timeout=30,
        default_headers={
            "HTTP-Referer": APP_URL,
            "X-Title": APP_NAME,
        }
    )

def get_openrouter_client():
    """
    获取OpenRouter客户端（直接使用）
    返回：配置好的OpenAI客户端
    """
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )