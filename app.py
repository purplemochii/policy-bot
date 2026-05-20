# aspp.py

import chromadb
import gradio as gr
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "BAAI/bge-small-en-v1.5"
)

# mistral connect
Settings.llm = Ollama(
    model = "phi3:mini",
    request_timeout = 600.0,
    context_window = 2048,
    system_prompt = "you are a policy assistant. answer questions based ONLY on the provided policy documents. If the answer is not in the documents, say so clearly. Do NOT make up information or answer from general knowledge."
)

# load index
db = chromadb.PersistentClient(path = "./chroma_db")
collection = db.get_or_create_collection("policy_docs")
vector_store = ChromaVectorStore(chroma_collection = collection)
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(similarity_top_k = 2)

def ask(question, history):
    if not question.strip():
        return ""
    response = query_engine.query(question)
    return str(response)

demo = gr.ChatInterface(
    fn = ask,
    title = "Collective brain",
    description = "Ask questions about course",
    examples = [
        "what is process management?",
        "explain the differences between SDN and traditional networking",
        "what is the CIA triad?"
    ]
)

demo.launch(share = True)