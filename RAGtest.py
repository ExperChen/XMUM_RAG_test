# 更新导入：使用新的 langchain-unstructured 包
from langchain_unstructured import UnstructuredLoader  # 更新的导入
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
# 使用ollama时用
# from langchain_ollama import OllamaLLM 
from langchain.chains import RetrievalQA
import warnings
import sys
import os
import io
from config import get_openrouter_langchain_model

# 强制 Python 输出中文时使用 utf-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 设置环境编码（Windows系统）
os.environ['PYTHONIOENCODING'] = 'utf-8'
# 精确过滤特定的警告信息
warnings.filterwarnings(
    "ignore", 
    message=".*encoder_attention_mask.*is deprecated.*",
    category=FutureWarning
)

# if sys.platform.startswith('win'):
#     # Windows 系统特殊处理
#     import locale
#     locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')


# 可选：过滤特定的警告信息
# warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. 加载本地文档
print("正在加载文档...")
# 使用新的 UnstructuredLoader
loader = UnstructuredLoader(r"test_docs\test.txt")
docs = loader.load()
print(f"加载了 {len(docs)} 个文档")

# 2. 切片
print("正在切片文档...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)
print(f"切片后共得到 {len(splits)} 个文档片段")

# 3. 过滤复杂元数据
filtered_splits = filter_complex_metadata(splits)
print(f"过滤元数据后有 {len(filtered_splits)} 个有效片段")

# 4. 创建向量数据库
print("正在创建向量数据库...")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(filtered_splits, embedding)
print("向量数据库创建成功！")

# 5. 初始化OpenRouter模型
print("正在初始化OpenRouter模型...")
# 第62行：不指定模型，使用默认
llm = get_openrouter_langchain_model()  # 使用config.py中的默认模型
# 或者如果需要特定模型：
# llm = get_openrouter_langchain_model("其他模型名")
print("OpenRouter模型初始化成功！")

# 6. 构建问答链
print("正在构建问答链...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=vectorstore.as_retriever(),
    return_source_documents = True
)

# 7. 运行问答
query = "请用中文总结这份文档主要讲了什么？"
print(f"\n问题：{query}")
print("正在生成回答...")

try:
    # 确保查询字符串正确编码
    query_encoded = query.encode('utf-8').decode('utf-8')
    
    # 调用问答链
    response = qa_chain.invoke({"query": query_encoded})
    
    # 确保结果正确解码
    if isinstance(response['result'], bytes):
        result = response['result'].decode('utf-8')
    else:
        result = response['result']
    
    print(f"\n回答：{result}")
    
    # 显示相关文档片段
    if 'source_documents' in response:
        print("\n相关文档片段：")
        for i, doc in enumerate(response['source_documents'][:2]):
            content = doc.page_content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            print(f"片段 {i+1}: {content[:200]}...")
            
except Exception as e:
    print(f"调用API时发生错误：{str(e)}")
    print("错误详情：")
    import traceback
    traceback.print_exc()
    print("请检查API密钥和网络连接")
