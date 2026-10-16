from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi

def hybrid_search(splited_chunks):
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("dp_dp_legal_documents")

    texts=[]
    ids=[]
    metadatas=[]

    for i,chunk in enumerate(splited_chunks):

        texts.append(chunk.page_content)

        doc_items=chunk.metadata.get("dl_meta",{}).get("doc_items",[])
        page_no=doc_items[0].get("prov",[{}])[0].get("page_no") if doc_items else 0

        metadatas.append({"page_number": int(page_no)})
        ids.append(f"chunk={i}")

    # for vector store ingestion
    print(f"Generating embeddings for {len(texts)} chunks.")
    embeddings=embedding_model.encode(texts)
    collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    print(f"Successfully stored {len(texts)} chunks with their flat metadatas.")

    # for keyword store ingestion
    print("Building BM25 keyword index ")
    # bm25 requires tokenized list of words and not strings
    tokenized_list=[text.lower().split(" ") for text in texts]
    bm25_index=BM25Okapi(tokenized_list)

    return bm25_index,texts,metadatas,ids,collection,embedding_model