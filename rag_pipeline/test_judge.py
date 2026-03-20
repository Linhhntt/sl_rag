import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Setup local LLM
llm = ChatOpenAI(
    base_url=f"{os.getenv('OLLAMA_BASE_URL')}/v1",
    api_key="ollama",
    model=os.getenv("GENERATION_MODEL")
)

# Setup judge model for evaluation
judge_llm = ChatOpenAI(
    base_url=f"{os.getenv('OLLAMA_BASE_URL')}/v1",
    api_key="ollama",
    model=os.getenv("JUDGE_MODEL")
)

# Load the DB
embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
vector_db = Chroma(collection_name="rag_collection", persist_directory="./chroma_db", embedding_function=embeddings)

# Simple RAG Chain
def query_rag(query, return_context=False):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    
    result = qa_chain.invoke(query)
    
    if return_context:
        # Get the retrieved documents for evaluation from the result
        source_docs = result.get("source_documents", [])
        context_docs = "\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(source_docs)])
        return result["result"], context_docs
    
    return result["result"]

    if return_context:
        # Get the retrieved documents for evaluation
        docs = retriever.get_relevant_documents(query)
        context_docs = "\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
        return result["result"], context_docs

    return result["result"]

# Judge model evaluation function
def evaluate_response(query, response, context_docs=None):
    """
    Evaluate the RAG response using Prometheus-2 judge model
    """
    if context_docs:
        template = """You are an expert evaluator assessing the quality of a RAG (Retrieval-Augmented Generation) system's response.

Please evaluate the following response based on these criteria:
1. Faithfulness: Does the response accurately reflect the information in the retrieved documents? Is it factually correct?
2. Relevance: Does the response directly address the user's query? Is it on-topic?
3. Helpfulness: Is the response useful and informative? Does it provide clear, actionable information?
4. Completeness: Does the response fully answer the query or are there important gaps?

Query: {query}
Response: {response}

Retrieved Context Documents:
{context_docs}

Provide a detailed evaluation with scores (1-5) for each criterion and an overall assessment.
Format your response as:
- Faithfulness: [score]/5 - [brief explanation]
- Relevance: [score]/5 - [brief explanation]
- Helpfulness: [score]/5 - [brief explanation]
- Completeness: [score]/5 - [brief explanation]
- Overall Assessment: [summary]"""
    else:
        template = """You are an expert evaluator assessing the quality of a RAG (Retrieval-Augmented Generation) system's response.

Please evaluate the following response based on these criteria:
1. Faithfulness: Does the response accurately reflect the information in the retrieved documents? Is it factually correct?
2. Relevance: Does the response directly address the user's query? Is it on-topic?
3. Helpfulness: Is the response useful and informative? Does it provide clear, actionable information?
4. Completeness: Does the response fully answer the query or are there important gaps?

Query: {query}
Response: {response}

Provide a detailed evaluation with scores (1-5) for each criterion and an overall assessment.
Format your response as:
- Faithfulness: [score]/5 - [brief explanation]
- Relevance: [score]/5 - [brief explanation]
- Helpfulness: [score]/5 - [brief explanation]
- Completeness: [score]/5 - [brief explanation]
- Overall Assessment: [summary]"""

    evaluation_prompt = PromptTemplate.from_template(template)
    evaluation_chain = evaluation_prompt | judge_llm
    
    params = {
        "query": query,
        "response": response
    }
    if context_docs:
        params["context_docs"] = context_docs
    
    evaluation = evaluation_chain.invoke(params)
    
    return evaluation.content

if __name__ == "__main__":
    # Test query
    user_query = "What is RAG and how does it work?"
    print(f"Query: {user_query}")
    print("\n--- RAG Response ---\n")
    response, context_docs = query_rag(user_query, return_context=True)
    print(response)

    print("\n--- Judge Model Evaluation ---\n")
    evaluation = evaluate_response(user_query, response, context_docs)
    print(evaluation)