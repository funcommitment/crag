from sentence_transformers import CrossEncoder
from collections import defaultdict

reranker = CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base")
def get_top_chunks(query: str, bm25_index,bm25_texts,bm25_metadatas,bm25_ids,collection,embedding_model,top_k: int = 3):
    ques_embedding = embedding_model.encode(query).tolist()
    que_result = collection.query(
        query_embeddings=[ques_embedding],
        n_results=10,
        include=["documents", "metadatas"]
    )
    # extracting top 10 result ids from vector search
    vector_hits=que_result["ids"][0] if que_result["ids"] else []

    #keyword search bm25
    tokenized_query=query.lower().split(" ")
    bm25_scores=bm25_index.get_scores(tokenized_query)

    bm25_ranked=sorted(zip(bm25_ids,bm25_scores),key=lambda x:x[1],reverse=True)[:10]
    bm25_hits=[hit[0] for hit in bm25_ranked]

    # reciprocal rank fusion step 
    rrf_scores=defaultdict(float)
    k_constant=60

    for rank,doc_id in enumerate(vector_hits):
        rrf_scores[doc_id]+=1 / (k_constant + rank + 1)

    for rank,doc_id in enumerate(bm25_hits):
        rrf_scores[doc_id]+=1 / (k_constant + rank + 1)

    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    #cross encoder(reranking)
    cross_inp=[]
    for doc_id, rrf_score in fused_results:
        idx = bm25_ids.index(doc_id) if doc_id in bm25_ids else -1
        chunk_text = bm25_texts[idx]
        cross_inp.append([query, chunk_text])

    rerank_scores=reranker.predict(cross_inp)
    final_reranked = sorted(zip(fused_results, rerank_scores), key=lambda x: x[1], reverse=True)

    results=[]
    for (doc_id, rrf_score), cross_score in final_reranked[:top_k]:
        idx = bm25_ids.index(doc_id) if doc_id in bm25_ids else -1
        results.append({
                "text": bm25_texts[idx],
                "page": bm25_metadatas[idx].get("page_number"),
                "score": float(cross_score)
            })

    return results