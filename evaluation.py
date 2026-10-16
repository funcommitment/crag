import pandas as pd
import time
from main import run_crag_agent, call_llm
from loaders import load_pdf
from embeddings import hybrid_search

df = pd.read_csv("native_synthetic_benchmark.csv")
results = []
splitted_chunks=load_pdf("DPDP_Rules_2025_English_only.pdf")
bm25_index,texts,metadatas,ids,collection,embedding_model=hybrid_search(splitted_chunks)
print(f"Starting Upgraded Native Evaluation on {len(df)} questions...")

for index, row in df.iterrows():
    question = row['question']
    ground_truth = row['ground_truth']
    
    print(f"\n[Q{index + 1}/{len(df)}] {question[:60]}...")

    #baseline llm 
    baseline_prompt = f"Answer this legal question based on the Indian DPDP Rules: {question}"
    llm_only_answer = call_llm(baseline_prompt, json_mode=False)
    
    #crag answer
    crag_output,context_blocks = run_crag_agent(question,bm25_index,texts,metadatas,ids,collection,embedding_model,3)
    rag_answer = crag_output[0] if isinstance(crag_output, tuple) else crag_output
    
    if not rag_answer:
        rag_answer = "Not found in the provided context."
        
    judge_prompt = f"""You are a strict Legal AI Assessor evaluating a Retrieval-Augmented Generation (RAG) pipeline.
    
Question: {question}
Ground Truth (Source Fact): {ground_truth}
RAG Answer: {rag_answer}

Analyze the RAG Answer against the Ground Truth and provide a verdict in EXACTLY this format:
**Properly answers the question?** [Yes/No] - [Brief reasoning]
**Does it hallucinate?** [Yes/No] - [Brief reasoning]
**Does it answer directly from the source?** [Yes/No] - [Brief reasoning]
"""
    
    verifier_verdict = call_llm(judge_prompt, json_mode=False)
    
    results.append({
        "id": index + 1,
        "category": "synthetic",
        "question": question,
        "ground_truth": ground_truth,
        "llm_only_answer": llm_only_answer,
        "rag_answer": rag_answer,
        "verifier_verdict": verifier_verdict
    })
    
    time.sleep(3)

results_df = pd.DataFrame(results)
results_df.to_csv("final_crag_benchmark_results.csv", index=False)
print("\nEvaluation Complete! Saved to final_crag_benchmark_results.csv")