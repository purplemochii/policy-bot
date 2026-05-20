# aspp.py

import chromadb
import gradio as gr
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# embed model
Settings.emned_model = HuggingFaceEmbedding(
    model_name = "BAAI/bge-small-en-v1.5"
)

# mistral connect
Settings.llm = Ollama(
    model = "mistral",
    request_timeout = 120.0,
    system_prompt = "you are a policy assistant. answer questions based ONLY on the provided policy documents. If the answer is not in the documents, say so clearly. Do NOT make up information or answer from general knowledge."
)

# load index
db = chromadb.PersistentClient(path = "./chroma_db")
collection = db.get_or_create_collection("policy_docs")
vector_store = ChromaVectorStore(chroma_collection = collection)
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(similarity_top_k = 5)

def ask(question):
    if not question.strip():
        return ""
    response = query_engine.query(question)
    return str(response)

demo = gr.ChatInterface(
    fn = ask,
    title = "Policy Assistant",
    description = "Ask questions about company policies.",
    examples = [
        "What is the leave policy?",
        "What is the process for raising a ticket?",
        "Whst are the data protection obligations for staff?"
    ]
)

demo.launch(share = True)