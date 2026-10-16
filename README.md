# Corrective RAG (CRAG) for Regulatory Legal Text

A production-grade Corrective Retrieval-Augmented Generation (CRAG) pipeline designed to query the Indian Digital Personal Data Protection (DPDP) Act with zero hallucinations. 

This system moves beyond basic vector retrieval by implementing an agentic verification loop. It grades its own context, rewrites failed queries using HyDE (Hypothetical Document Expansion), and forces a graceful fallback if the exact statutory clause cannot be found in the index.

**Current Benchmark:** 28/28 (100% Verified Retrieval) using `openai/gpt-oss-120b`.

## 🧠 Core Engineering Decisions & Architecture

### 1. Adversarial Synthetic Test Suite (`generate_testset.py`)
Evaluating AI on simple factoid questions is a solved problem. To truly stress-test the retrieval system, I wrote a dedicated Python pipeline that forces a 120B frontier model to generate highly complex, multi-hop legal queries (e.g., cross-referencing nested clauses and definitions). 
* **Why:** Test suites should reflect real-world edge cases. Generating an adversarial benchmark ensures the pipeline doesn't just find keywords, but actually understands and navigates legal semantics.

### 2. Custom Eval Harness & LLM-as-a-Judge (`evaluation.py`)
Built a ground-truth evaluation harness that mixes deterministic execution tracking with an LLM-as-a-judge scorer. 
* **Why:** Standard RAG metrics (like cosine similarity) are useless for legal text where a single word changes the ruling. The 120B judge evaluates the final agent output strictly against the synthetic ground truth, logging clear pass/fail verdicts and catching prompt regressions during iteration.

### 3. Agentic Tracing & Deterministic JSON Sanitization (`main.py`)
The pipeline runs a continuous trace: Grader → Rewriter → Search → Generator. 
* **The Bug:** During trace logging, I caught a regression where the Rewriter model leaked raw JSON brackets into the BM25 sparse index, causing a retrieval crash (`NoneType` tuple unpacking). 
* **The Fix:** Implemented deterministic `try/except` JSON parsing between agent handoffs. The code now sanitizes the LLM's output before hitting the vector database, ensuring structural model regressions never silently break the search.

### 4. Generalized for Legal, Adaptable to Anything
While benchmarked on the DPDP Act, this architecture natively supports *any* dense legal or regulatory corpus out of the box. 
* **Why it matters:** The agentic verification loop (Hybrid Search → Grader → HyDE → Fallback) is entirely decoupled from the data. Because it relies on systemic logic rather than hardcoded rules, adapting this zero-hallucination pipeline to entirely different high-stakes domains—like preventive healthcare datasets or financial compliance—requires only swapping the ingested PDFs and basic system prompt tweaks.

## 🛠 Tech Stack
* **Language & Orchestration:** Python, LangChain, Groq API (`openai/gpt-oss-120b`)
* **Retrieval:** Hybrid Search (ChromaDB Dense Vectors + BM25 Sparse Indexing)
* **Testing:** Automated LLM-as-a-Judge Evaluation Loop
* **Frontend:** Streamlit

## 💻 Quick Start

*Note: The frontend is currently configured for local execution. Public cloud deployment will be live shortly.*

```bash
# Clone repository
git clone [https://github.com/funcommitment/legal-crag.git](https://github.com/funcommitment/legal-crag.git)
cd legal-crag

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY to the .env file

# Run the automated eval harness
python evaluation.py

# Launch the interactive Streamlit UI
streamlit run app.py
