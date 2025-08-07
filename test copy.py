from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="deepseek-r1:8b",  # 模型名称与你 pull 的一致
    temperature=0.7
)

print(llm.invoke("LangChain是什么"))
