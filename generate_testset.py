import json
import time
import pandas as pd
from loaders import load_pdf
from main import call_llm

raw_chunks = load_pdf("DPDP_Rules_2025_English_only.pdf")
synthetic_data = []

print("Generating synthetic questions using native Groq client...")

for i, chunk in enumerate(raw_chunks[:15]):
    text = chunk.page_content.strip()
    
    if len(text) < 150:
        continue
        
    print(f"Generating tricky questions for chunk {i+1}...")
    
    # Prompt Fencing: Keep the text, but control the LLM's focus
    prompt = f"""You are a senior legal auditor analyzing the Indian DPDP Rules. 
Read the context below. Use the headers and titles to understand the topic, but write 2 tricky questions based ONLY on the deep regulatory rules, penalty conditions, or compliance constraints found in the body text.

STRICT RULES:
1. DO NOT write questions asking "What is the title of..." or "When was this published?".
2. The questions must require reading the complex clauses to answer.
3. Output strictly as JSON: {{"qas": [{{"question": "...", "ground_truth": "..."}}]}}

Context: 
{text}"""

    response = call_llm(prompt, json_mode=True)
    
    try:
        data = json.loads(response)
        for qa in data.get("qas", []):
            synthetic_data.append({
                "question": qa["question"],
                "ground_truth": qa["ground_truth"]
            })
    except Exception as e:
        print(f"Failed to parse JSON for chunk {i}: {e}")

    time.sleep(2) 

df = pd.DataFrame(synthetic_data)
df.to_csv("native_synthetic_benchmark.csv", index=False)
print(f"\nSuccess! Saved {len(synthetic_data)} high-quality legal questions.")