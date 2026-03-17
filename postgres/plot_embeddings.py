import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sklearn.manifold import TSNE
from vector_db.store_to_pgvector import get_vector_store

def get_query_plot_data(collection_name, user_query, retrieved_docs):
    vector_store = get_vector_store(collection_name)
    # 1. Get query embedding
    query_embedding = vector_store.embeddings.embed_query(user_query)
    # 2. Get all embeddings from DB
    with Session(vector_store.session_maker.bind) as session:
        db_results = (
            session.query(vector_store.EmbeddingStore.embedding, vector_store.EmbeddingStore.document)
            .join(vector_store.CollectionStore)
            .filter(vector_store.CollectionStore.name == collection_name)
            .all()
        )
    all_embeddings = [r[0] for r in db_results]
    all_texts = [r[1] for r in db_results]
    # 3. Combine: DB points + User Query point
    combined_embeddings = np.vstack([all_embeddings, query_embedding])
    # 4. Dimensionality Reduction (1536 -> 2)
    # Use PCA if t-SNE is too slow for your dataset size
    perp = min(30, len(combined_embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    reduced = tsne.fit_transform(combined_embeddings)
    # 5. Build DataFrame for plotting
    df = pd.DataFrame(reduced, columns=["x", "y"])
    df["label"] = "Other Chunks"
    df["snippet"] = all_texts + ["USER QUERY"]
    # Mark the user query (last item)
    df.iloc[-1, df.columns.get_loc("label")] = "Current Query"
    # Highlight retrieved docs (match by snippet/content)
    retrieved_contents = [d.page_content for d in retrieved_docs]
    df.loc[df["snippet"].isin(retrieved_contents), "label"] = "Retrieved Context"
    return df
