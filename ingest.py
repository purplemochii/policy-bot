# ingest.py

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name = "BAAI/bge-small-en-v1.5"
)

Settings.llm = None

documents = SimpleDirectoryReader(input_dir = "docs").load_data()
print(f"loaded {len(documents)} document chunks")

db = chromadb.PersistentClient(path = "./chroma_db")
collection = db.get_or_create_collection("policy_docs")
vector_store = ChromaVectorStore(chroma_collection = collection)
storage_context = StorageContext.from_defaults(vector_store = vector_store)

# build index and persist
index = VectorStoreIndex.from_documents(
    documents, storage_context = storage_context
)

print("done. :)")