"""
RAG系统的Gradio可视化界面
功能：文件上传、文档切片查看、智能问答
"""

import gradio as gr
import os
import tempfile
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from config import get_langchain_model
import warnings

# 过滤警告信息
warnings.filterwarnings("ignore", category=FutureWarning)

# 全局变量存储当前的RAG系统组件
current_vectorstore = None
current_qa_chain = None
current_splits = []

def process_uploaded_file(file_path, chunk_size, chunk_overlap, split_method):
    """
    处理上传的文件：加载、切片、创建向量数据库
    
    参数:
        file_path: 上传文件的路径
        chunk_size: 切片大小
        chunk_overlap: 切片重叠大小
        split_method: 切片方法（"按字符数" 或 "按段落"）
    
    返回:
        处理状态信息和切片内容
    """
    global current_vectorstore, current_qa_chain, current_splits
    
    try:
        if file_path is None:
            return "请先上传文件！", ""
        
        # 1. 加载文档
        status = "正在加载文档...\n"
        loader = UnstructuredLoader(file_path)
        docs = loader.load()
        status += f"✅ 成功加载 {len(docs)} 个文档片段\n"
        
        # 2. 切片文档
        status += "正在切片文档...\n"
        
        if split_method == "按字符数":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        else:  # 按段落
            splitter = CharacterTextSplitter(
                separator="\n", 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        
        splits = splitter.split_documents(docs)
        current_splits = splits  # 保存切片结果
        status += f"✅ 切片完成，共得到 {len(splits)} 个文档片段\n"
        
        # 3. 过滤元数据
        filtered_splits = filter_complex_metadata(splits)
        status += f"✅ 过滤元数据后有 {len(filtered_splits)} 个有效片段\n"
        
        # 4. 创建向量数据库
        status += "正在创建向量数据库...\n"
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        current_vectorstore = Chroma.from_documents(filtered_splits, embedding)
        status += "✅ 向量数据库创建成功！\n"
        
        # 5. 初始化问答链
        status += "正在初始化问答系统...\n"
        llm = get_langchain_model()
        current_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=current_vectorstore.as_retriever(),
            return_source_documents=True
        )
        status += "✅ 问答系统初始化成功！\n\n🎉 文档处理完成，现在可以开始提问了！"
        
        # 生成切片内容预览
        chunks_preview = generate_chunks_preview(splits)
        
        return status, chunks_preview
        
    except Exception as e:
        error_msg = f"❌ 处理文件时发生错误：{str(e)}"
        return error_msg, ""

def generate_chunks_preview(splits):
    """
    生成切片内容的预览文本
    
    参数:
        splits: 文档切片列表
    
    返回:
        格式化的切片预览文本
    """
    preview = "=== 文档切片内容预览 ===\n\n"
    
    for i, split in enumerate(splits[:10]):  # 只显示前10个切片
        preview += f"--- 切片 {i + 1} ---\n"
        preview += f"长度：{len(split.page_content)} 字符\n"
        preview += f"内容：{split.page_content[:200]}...\n"
        preview += "-" * 50 + "\n\n"
    
    if len(splits) > 10:
        preview += f"... 还有 {len(splits) - 10} 个切片未显示 ...\n"
    
    return preview

def answer_question(question, show_full_content=True):
    """
    回答用户问题
    
    参数:
        question: 用户提出的问题
        show_full_content: 是否显示完整的文档片段内容
    
    返回:
        问答结果和相关文档片段
    """
    global current_qa_chain
    
    if current_qa_chain is None:
        return "❌ 请先上传并处理文档！", ""
    
    if not question.strip():
        return "❌ 请输入问题！", ""
    
    try:
        # 调用问答链
        response = current_qa_chain.invoke({"query": question})
        
        # 获取回答
        answer = response['result']
        
        # 获取相关文档片段
        source_docs = ""
        if 'source_documents' in response and response['source_documents']:
            source_docs = "\n=== 相关文档片段 ===\n\n"
            for i, doc in enumerate(response['source_documents'][:3]):
                content = doc.page_content
                
                if show_full_content:
                    # 显示完整内容
                    source_docs += f"片段 {i+1}（长度：{len(content)} 字符）：\n"
                    source_docs += f"{content}\n\n"
                else:
                    # 显示截断内容
                    source_docs += f"片段 {i+1}：\n{content[:300]}...\n\n"
                
                source_docs += "-" * 50 + "\n\n"
        
        return answer, source_docs
        
    except Exception as e:
        error_msg = f"❌ 回答问题时发生错误：{str(e)}"
        return error_msg, ""

def view_all_chunks():
    """
    查看所有切片内容
    
    返回:
        完整的切片内容
    """
    global current_splits
    
    if not current_splits:
        return "❌ 没有可显示的切片内容，请先上传并处理文档！"
    
    full_content = "=== 完整切片内容 ===\n\n"
    
    for i, split in enumerate(current_splits):
        full_content += f"--- 切片 {i + 1} ---\n"
        full_content += f"长度：{len(split.page_content)} 字符\n"
        full_content += f"内容：\n{split.page_content}\n"
        full_content += "=" * 60 + "\n\n"
    
    return full_content

# 创建Gradio界面
def create_interface():
    """
    创建Gradio用户界面
    """
    with gr.Blocks(title="RAG智能问答系统", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 📚 RAG智能问答系统
            
            欢迎使用基于检索增强生成(RAG)的智能问答系统！
            
            ## 使用步骤：
            1. 📁 上传文档文件（支持PDF、TXT、DOCX等格式）
            2. ⚙️ 设置切片参数并处理文档
            3. 👀 查看文档切片内容
            4. ❓ 向系统提问，获得基于文档的智能回答
            """
        )
        
        with gr.Tab("📁 文档上传与处理"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 文件上传组件
                    file_upload = gr.File(
                        label="上传文档文件",
                        file_types=[".pdf", ".txt", ".docx", ".md"],
                        type="filepath"
                    )
                    
                    # 切片参数设置
                    gr.Markdown("### ⚙️ 切片参数设置")
                    chunk_size = gr.Slider(
                        minimum=100, 
                        maximum=2000, 
                        value=500, 
                        step=50,
                        label="切片大小（字符数）"
                    )
                    chunk_overlap = gr.Slider(
                        minimum=0, 
                        maximum=200, 
                        value=50, 
                        step=10,
                        label="切片重叠（字符数）"
                    )
                    split_method = gr.Radio(
                        choices=["按字符数", "按段落"],
                        value="按字符数",
                        label="切片方法"
                    )
                    
                    # 处理按钮
                    process_btn = gr.Button("🚀 处理文档", variant="primary")
                
                with gr.Column(scale=2):
                    # 处理状态显示
                    process_status = gr.Textbox(
                        label="处理状态",
                        lines=10,
                        interactive=False
                    )
        
        with gr.Tab("👀 切片内容查看"):
            with gr.Row():
                view_chunks_btn = gr.Button("📋 查看所有切片", variant="secondary")
            
            chunks_display = gr.Textbox(
                label="文档切片内容",
                lines=20,
                interactive=False,
                show_copy_button=True
            )
        
        with gr.Tab("❓ 智能问答"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="例如：这份文档主要讲了什么？",
                        lines=3
                    )
                    ask_btn = gr.Button("🤔 提问", variant="primary")
                
                with gr.Column(scale=3):
                    answer_output = gr.Textbox(
                        label="AI回答",
                        lines=10,
                        interactive=False,
                        show_copy_button=True
                    )
            
            source_docs_output = gr.Textbox(
                label="相关文档片段",
                lines=8,
                interactive=False,
                show_copy_button=True
            )
        
        # 绑定事件
        process_btn.click(
            fn=process_uploaded_file,
            inputs=[file_upload, chunk_size, chunk_overlap, split_method],
            outputs=[process_status, chunks_display]
        )
        
        view_chunks_btn.click(
            fn=view_all_chunks,
            outputs=chunks_display
        )
        
        ask_btn.click(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, source_docs_output]
        )
        
        # 支持回车键提问
        question_input.submit(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, source_docs_output]
        )
    
    return app

# 启动应用
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",  # 允许外部访问
        server_port=7860,       # 端口号
        share=True,            # 是否创建公共链接
        debug=True              # 调试模式
    )