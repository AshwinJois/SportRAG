import re
from langchain_openai import ChatOpenAI
from postgres.connection import fetch_documents, insert_evaluation_data

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
def generate_qa_from_chunk(chunk, num_qas=3):
    """
    Generate multiple question-answer pairs from a single chunk.
    """
    prompt = f"""
You are generating evaluation data for a RAG system.
From the context below, generate {num_qas} diverse question-answer pairs.
Context:
{chunk}
Return format:
Q1: ...
A1: ...

Q2: ...
A2: ...

Q3: ...
A3: ...
"""
    response = llm.invoke(prompt).content
    return response

def parse_qa_pairs(text):
    """
    Parse multiple QA pairs from LLM output.
    """
    pattern = r"Q\d+:\s*(.*?)\nA\d+:\s*(.*?)(?=\nQ\d+:|\Z)"
    matches = re.findall(pattern, text, re.S)
    return [(q.strip(), a.strip()) for q, a in matches]

def generate_synthetic_dataset(target_count=100):
    """
    Generate synthetic dataset until reaching target_count of QA pairs.
    """
    documents = fetch_documents()
    total_inserted = 0
    for doc in documents:
        try:
            response = generate_qa_from_chunk(doc)
            qa_pairs = parse_qa_pairs(response)
            for question, answer in qa_pairs:
                insert_evaluation_data(question, answer, doc)
                total_inserted += 1
                if total_inserted >= target_count:
                    return total_inserted
        except Exception as e:
            print("Error processing chunk:", e)
    return total_inserted