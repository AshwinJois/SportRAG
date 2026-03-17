import os
import openai
from dotenv import load_dotenv
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings # Or your preferred embedding model
from postgres.connection import connect_to_db

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# 1. Initialize your embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small")
# 2. Connect to Database
connection = connect_to_db()
# Used during ingestion
def store_to_db(all_documents, collection_name: str):
    vector_store = PGVector.from_documents(
        documents=all_documents, # These are your chunks from the previous step
        embedding=embeddings,
        connection=connection,
        collection_name=collection_name,
        use_jsonb=True, # Stores metadata in a flexible JSONB column
    )
    return vector_store

# Used during querying
def get_vector_store(collection_name: str):  # ← accepts name now
    vector_store = PGVector(
        connection=connection,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return vector_store