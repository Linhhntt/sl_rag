import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

load_dotenv()

# Setup local LLM
llm = ChatOpenAI(
    base_url=f"{os.getenv('OLLAMA_BASE_URL')}/v1",
    api_key="ollama",
    model=os.getenv("GENERATION_MODEL")
)

# Load the DB
embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Simple RAG Chain
def query_rag(query):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    result = qa_chain.invoke(query)
    return result["result"]

if __name__ == "__main__":
    user_query = input("Ask a question: ")
    print("\n--- Response ---\n")
    print(query_rag(user_query))