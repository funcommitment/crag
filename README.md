# Corrective RAG (CRAG) for Regulatory Legal Text

A production-grade Corrective Retrieval-Augmented Generation (CRAG) pipeline designed to query the Indian Digital Personal Data Protection (DPDP) Act with zero hallucinations. 

This system moves beyond basic vector retrieval by implementing an agentic verification loop. It grades its own context, rewrites failed queries using HyDE (Hypothetical Document Expansion), and forces a graceful fallback if the exact statutory clause cannot be found in the index.

**Current Benchmark:** 28/28 (100% Verified Retrieval) using `openai/gpt-oss-120b`.

## 🧠 Core Engineering Decisions & Architecture

### 1. Hybrid Search & Cross-Encoder Reranking
Legal queries require both semantic understanding and exact keyword matching (e.g., citing "Section 40(1)"). 
* **The Pipeline:** The system uses ChromaDB for dense vector search (semantics) fused with a BM25 sparse index (exact alphanumeric matching). 
* **The Reranker:** Because hybrid search casts a wide net, the merged results are passed through a Cross-Encoder reranker. This computes direct attention between the user's query and each chunk, pushing the most highly correlated context to the top `k` before it ever hits the LLM context window.

### 2. Adversarial Synthetic Test Suite (`generate_testset.py`)
Evaluating AI on simple factoid questions is a solved problem. To truly stress-test the retrieval system, I wrote a dedicated Python pipeline that forces a 120B frontier model to generate highly complex, multi-hop legal queries (e.g., cross-referencing nested clauses and definitions). 
* **Why:** Test suites should reflect real-world edge cases. Generating an adversarial benchmark ensures the pipeline doesn't just find keywords, but actually understands and navigates complex legal semantics.

### 3. Custom Eval Harness & LLM-as-a-Judge (`evaluation.py`)
Built a ground-truth evaluation harness that mixes deterministic execution tracking with an LLM-as-a-judge scorer. 
* **Why:** Standard RAG metrics (like cosine similarity) are useless for legal text where a single word changes the ruling. The 120B judge evaluates the final agent output strictly against the synthetic ground truth, logging clear pass/fail verdicts and catching prompt regressions during iteration.

### 4. Agentic Tracing & Deterministic JSON Sanitization (`main.py`)
The pipeline runs a continuous trace: Grader → Rewriter → Search → Generator. 
* **The Bug:** During trace logging, I caught a regression where the Rewriter model leaked raw JSON brackets into the BM25 sparse index, causing a retrieval crash (`NoneType` tuple unpacking). 
* **The Fix:** Implemented deterministic `try/except` JSON parsing between agent handoffs. The code now sanitizes the LLM's output before hitting the vector database, ensuring structural model regressions never silently break the search.

### 5. Generalized for Legal, Adaptable to Anything
While benchmarked on the DPDP Act, this architecture natively supports *any* dense legal or regulatory corpus out of the box. 
* **Why it matters:** The agentic verification loop (Hybrid Search → Grader → HyDE → Fallback) is entirely decoupled from the data. Because it relies on systemic logic rather than hardcoded rules, adapting this zero-hallucination pipeline to entirely different high-stakes domains—like preventive healthcare datasets or financial compliance—requires only swapping the ingested PDFs and basic system prompt tweaks.

### 6. Hierarchical Document Parsing (Docling)
Standard PDF loaders destroy structural formatting, which is fatal for legal documents reliant on strict parent-child clause relationships. 
* **The Implementation:** I used `DoclingLoader` to parse the DPDP Act. This preserves the hierarchical DOM of the document (headers, nested lists, tables) before text splitting. 
* **Why it matters:** It ensures that chunks are divided by semantic boundaries rather than arbitrary character counts, preventing fragmented retrieval and giving the LLM the exact structural context it needs to reason accurately.

## 🛠 Tech Stack
* **Language & Orchestration:** Python, LangChain, Groq API (`openai/gpt-oss-120b`)
* **Ingestion & Parsing:** docling (Hierarchical PDF Parsing)
* **Retrieval & Ranking:** ChromaDB (Dense Vectors), BM25 (Sparse Index), Cross-Encoder (Reranking)
* **Testing:** Automated LLM-as-a-Judge Evaluation Loop
* **Frontend:** Streamlit

## 💻 Quick Start

*Note: The frontend is currently configured for local execution. Public cloud deployment will be live shortly.*

```bash
# Clone repository
git clone [https://github.com/funcommitment/crag.git](https://github.com/funcommitment/crag.git)
cd crag

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY to the .env file

# Run the automated eval harness
python evaluation.py

# Launch the interactive Streamlit UI
streamlit run app.py
