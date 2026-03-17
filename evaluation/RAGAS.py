from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from postgres.connection import fetch_evaluation_df
from llm_model.model import response_generation_stream, build_agent
from postgres.connection import fetch_evaluation_rows, update_rag_response
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

def run_rag_evaluation():
    agent = build_agent()
    rows = fetch_evaluation_rows()  # only rows without rag_response
    for row_id, question, context in rows:
        # retrieved_docs expected to be list of document objects
        # we can fake a doc object with page_content = context
        class Doc:
            def __init__(self, page_content):
                self.page_content = page_content
                self.metadata = {}
        retrieved_docs = [Doc(context)]
        full_response = ""
        # Streaming RAG response
        for chunk in response_generation_stream(agent, retrieved_docs, question, thread_id=row_id):
            full_response += chunk
        # Save the RAG response in the table
        update_rag_response(row_id, full_response)
        print(f"✅ Row {row_id} updated")

def run_ragas_evaluation():
    df = fetch_evaluation_df()
    if df.empty:
        raise ValueError("No evaluation data found. Run the RAG pipeline first.")
    ragas_data = {
        "question":     df["question"].tolist(),
        "answer":       df["rag_response"].tolist(),
        "contexts":     [[ctx] for ctx in df["context"].tolist()],
        "ground_truth": df["ground_truth"].tolist(),
    }
    dataset = Dataset.from_dict(ragas_data)
    # Wrap properly for RAGAS
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    return result