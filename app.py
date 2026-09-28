from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0,
    max_retries=0,
)

print("Gemini connection setup successful!")

pdf_path = "documents/22.Administrative & HR Policy 09.09.2024.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print(f"PDF loaded successfully!")
print(f"Number of pages loaded: {len(documents)}")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Text splitting successful!")
print(f"Number of chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("Embeddings created successfully!")
print("FAISS vector store created successfully!")
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever created successfully!")


class HelpdeskState(TypedDict):
    question: str
    context: str
    answer: str
def retrieve_node(state: HelpdeskState):
    retrieved_docs = retriever.invoke(state["question"])

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    return {
        "context": context
    }
def generate_node(state: HelpdeskState):
    prompt = f"""
You are the MLRIT Student Helpdesk Assistant.

Answer the student's question using ONLY the provided context.

If the context does not contain enough information to answer,
say that the information is not available in the provided documents.

Student question:
{state["question"]}

Context:
{state["context"]}
"""

    response = llm.invoke(prompt)

    answer = response.content

    answer = response.content

    if isinstance(answer, list) and answer:
        answer = answer[0].get("text", "")
    elif not isinstance(answer, str):
        answer = str(answer)

    return {
        "answer": answer
    }

graph = StateGraph(HelpdeskState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

helpdesk_graph = graph.compile()

print("LangGraph compiled successfully!")

question = "What are the leave rules for faculty?"

result = helpdesk_graph.invoke({
    "question": question
})

print("\nFinal Answer:")
print(result["answer"])