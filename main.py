import json
from retrieval import get_top_chunks
import os
from dotenv import load_dotenv
from groq import Groq
from loaders import load_pdf
from embeddings import hybrid_search

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(prompt: str, json_mode: bool = False,model_name: str = "openai/gpt-oss-120b") -> str:
    try:
        response=client.chat.completions.create(
            model=model_name,
            messages=[
                    # System prompt forcing JSON mode compliance if active
                    {"role": "system", "content": "You are a legal AI. Always output valid JSON if requested."},
                    {"role": "user", "content": prompt}
                ],
            response_format={"type": "json_object"} if json_mode else None,
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[API Error] {e}")
        # Failsafe: If the API crashes, hand the while loop a safe 'False' to trigger a retry
        return '{"relevant": false}' if json_mode else "Error: Generation failed."

def run_crag_agent(user_query: str,bm25_index,texts,metadatas,ids,collection,embedding_model,max_retries: int = 3,ui=None):
    print(f"user query :{user_query}")
    def log(msg):
        print(msg)
        if ui: ui.write(msg)

    log(f"User query: {user_query}")
    current_query=user_query
    attempt=1
    while attempt<=max_retries:
        log(f"Attempt: {attempt}/{max_retries}")
        top_chunks=get_top_chunks(current_query,bm25_index,texts,metadatas,ids,collection,embedding_model,3)
        log(f"searching for this query :{current_query}")
        
        context_blocks=[]
        for i,chunk in enumerate(top_chunks):
            context_blocks.append(f"--- Document {i+1} (Page {chunk['page']}) ---\n{chunk['text']}\n")
        final_context = "\n".join(context_blocks)
        grader_prompt = f"""You are a strict legal verification grader.
Evaluate if the following context contains the exact factual answer to the question.
If YES, output ONLY: {{"relevant": true}}
If NO, output ONLY: {{"relevant": false}}

Context:
{final_context}

Question: {current_query}"""

        log("[Agent] Asking Grader to verify context...")
        # (Be sure to pass json_mode=True to your actual LLM client so it forces JSON output)
        grader_response = call_llm(grader_prompt, json_mode=True)
        try:
            is_relevant = json.loads(grader_response).get("relevant", False)
        except json.JSONDecodeError:
            is_relevant = False # Default to false if the LLM hallucinates the JSON format
        if is_relevant:
            log("[Agent] Context VERIFIED. Generating final answer...")
            generator_prompt = f"""You are an expert Indian Legal AI analyzing the DPDP Rules.
Answer the question accurately using ONLY the provided context.

Context:
{final_context}

Question: {user_query}"""
            
            final_answer = call_llm(generator_prompt)
            return final_answer, context_blocks
        
        else:
            log("[Agent] Context REJECTED. Triggering Retry Loop...")
            if attempt < max_retries:
                # 4. The Rewriter Node (HyDE - Hypothetical Document Expansion)
                log("[Agent] Rewriting query to expand search net...")
                rewriter_prompt = f"""You are a legal research Rewriter.
The original search query failed to find the specific legal statute.
Rewrite the following question into a declarative statement that sounds like a legal clause, to improve vector search retrieval.
Original Question: {user_query}"""
                
                raw_rewrite = call_llm(rewriter_prompt)
                try:
                    current_query = json.loads(raw_rewrite).get("rewritten_query", raw_rewrite)
                except json.JSONDecodeError:
                    current_query = raw_rewrite
            
            attempt += 1

    # 5. Graceful Fallback (If all 3 attempts fail)
    fallback_msg="I have searched the index 3 times, but this specific provision does not appear to be explicitly mentioned in the text. I cannot provide an answer without hallucinating."
    log("\n================ FINAL ANSWER ================\n")
    log(fallback_msg)
    log("\n==============================================")
    return fallback_msg,[]

