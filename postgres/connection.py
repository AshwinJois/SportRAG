import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

def get_sqlalchemy_url():
    """For LangChain PGVector (SQLAlchemy driver)"""
    return f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_psycopg2_url():
    """For raw psycopg2 queries"""
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def connect_to_db():
    connection = get_sqlalchemy_url()   # ← used by PGVector
    #collection_name = "medical_research_docs"
    return connection #, collection_name

def create_evaluation_table():
    conn = psycopg2.connect(get_psycopg2_url())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE evaluation_dataset (
                    id SERIAL PRIMARY KEY,
                    question TEXT,
                    ground_truth TEXT,
                    context TEXT
                );
            """)
            conn.commit()
    finally:
        conn.close()

def create_ragas_result_table():
    conn = psycopg2.connect(get_psycopg2_url())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE ragas_result (
                    id SERIAL PRIMARY KEY,
                    faithfulness FLOAT,
                    answer_relevancy FLOAT,
                    context_precision FLOAT,
                    context_recall FLOAT
                );
            """)
            conn.commit()
    finally:
        conn.close()


def fetch_documents(limit=None):
    """Fetch chunks from langchain_pg_embedding."""
    conn = psycopg2.connect(get_psycopg2_url())
    try:
        with conn.cursor() as cur:
            sql = "SELECT document FROM langchain_pg_embedding WHERE document IS NOT NULL"
            if limit:
                sql += f" LIMIT {limit}"
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

def insert_evaluation_data(question, ground_truth, context):
    conn = psycopg2.connect(get_psycopg2_url())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO evaluation_dataset (question, ground_truth, context)
                VALUES (%s, %s, %s);
            """, (question, ground_truth, context))
            conn.commit()
    finally:
        conn.close()

def insert_ragas_result(faithfulness, answer_relevancy, context_precision, context_recall):
    #create_ragas_result_table()
    conn = psycopg2.connect(get_psycopg2_url())
    try:
        with conn.cursor() as cur:
            # IF NOT EXISTS so it's safe to call every time
            cur.execute("""
                           CREATE TABLE IF NOT EXISTS ragas_result (
                               id SERIAL PRIMARY KEY,
                               faithfulness FLOAT,
                               answer_relevancy FLOAT,
                               context_precision FLOAT,
                               context_recall FLOAT,
                               evaluated_at TIMESTAMP DEFAULT NOW()
                           );
                       """)
            cur.execute("""
                INSERT INTO ragas_result (faithfulness, answer_relevancy, context_precision, context_recall)
                VALUES (%s, %s, %s, %s);
            """, (faithfulness, answer_relevancy, context_precision, context_recall))
            conn.commit()
    finally:
        conn.close()


def ensure_evaluation_table():
    conn = psycopg2.connect(get_psycopg2_url())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_dataset (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    ground_truth TEXT NOT NULL,
                    context TEXT
                );
            """)
            conn.commit()
    finally:
        conn.close()

def list_collections():
    conn = psycopg2.connect(get_psycopg2_url())  # ← plain postgresql:// for psycopg2
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM langchain_pg_collection ORDER BY name;")
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"Could not fetch collections: {e}")
        return []
    finally:
        conn.close()

def fetch_evaluation_rows():
    """Fetch all QA rows with question and context"""
    conn = psycopg2.connect(get_psycopg2_url())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, question, context
                FROM evaluation_dataset
                WHERE rag_response IS NULL;
            """)
            rows = cur.fetchall()  # [(id, question, context), ...]
            return rows
    finally:
        conn.close()

def update_rag_response(row_id, response_text):
    """Update the rag_response for a given row"""
    conn = psycopg2.connect(get_psycopg2_url())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE evaluation_dataset
                SET rag_response = %s
                WHERE id = %s;
            """, (response_text, row_id))
            conn.commit()
    finally:
        conn.close()

def fetch_evaluation_df():
    engine = create_engine(get_sqlalchemy_url())
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT question, ground_truth, context, rag_response
            FROM evaluation_dataset
            WHERE rag_response IS NOT NULL
        """, conn)
    return df

def fetch_ragas_results():
    engine = create_engine(get_sqlalchemy_url())
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT faithfulness, answer_relevancy, context_precision, context_recall
            FROM ragas_result
            ORDER BY id DESC
        """, conn)
    return df