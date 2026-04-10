# SmartMCI — Corrosion and Integrity Engineering Assistant

A domain-specific GenAI application built for Materials, Corrosion, and Integrity (MCI) engineers in the oil and gas industry. It combines a RAG pipeline over API standards documentation with a Groq-hosted LLM to answer engineering questions, calculate corrosion rates, and generate structured damage mechanism assessments.

Built from direct field experience — the developer worked as an offshore construction and maintenance project engineer and built this tool to show what applied GenAI looks like in an industrial engineering context.

---

## What it does

Three modes, each with a distinct purpose:

**Chatbot** — conversational Q&A grounded in API 571, 970, and 584 documents. Answers are retrieved from the vector store before the LLM generates a response, keeping the tool within the standards rather than hallucinating outside them. Tavily web search supplements with current information when needed.

**Calculator** — corrosion rate calculation (mm/year) and remaining equipment life prediction. Validates inputs against API 571 thresholds and generates an inspection report with trend projections and recommendations.

**Analysis** — structured damage mechanism assessment for a specific piece of equipment. Takes equipment type, material, operating conditions, and damage mechanism as inputs, and outputs a full assessment: damage probability, recommended inspection method, mitigation strategy, and operating limits.

---

## Architecture

**RAG ingestion pipeline (`ingestion_pc.py`)** — runs once to build the knowledge base:

1. Load API 571, 970, and 584 PDFs from `data/raw_documents/`
2. Split into 1000-token chunks with 200-token overlap (`RecursiveCharacterTextSplitter`)
3. Embed with `sentence-transformers/all-mpnet-base-v2` via HuggingFace
4. Store in three separate Pinecone serverless indexes — one per API standard

**At query time** — the app retrieves relevant chunks from the appropriate index, injects them into a structured prompt template via LangChain, and sends to the Groq API (Llama 3.1-8B).

---

## Stack

| Layer            | Tool                                                         |
| ---------------- | ------------------------------------------------------------ |
| UI               | Streamlit                                                    |
| LLM              | Groq API (Llama 3.1-8B)                                      |
| Orchestration    | LangChain                                                    |
| Vector database  | Pinecone (3 indexes: api571, api970, api584)                 |
| Embeddings       | HuggingFace — `sentence-transformers/all-mpnet-base-v2`      |
| Web search       | Tavily API                                                   |
| Input validation | Pydantic v2 with custom enums for equipment type, material, damage mechanism |
| Charts           | Plotly                                                       |

---

## What this demonstrates

- RAG pipeline end-to-end: document ingestion, chunking, embedding, vector store indexing, retrieval-augmented generation
- Domain-specific LLM application: prompt engineering for a specialized engineering context grounded in industry standards
- Pydantic v2 data validation: typed input models with custom enums prevent invalid engineering parameters from reaching LLM calls
- Multi-mode Streamlit application: chatbot, calculator, and analysis flows sharing the same knowledge base and LLM client

---

## Run locally

```bash
pip install -r requirements.txt
```

Set the following in `.env`:

```
GROQ_API_KEY=
PINECONE_API_KEY=
TAVILY_API_KEY=
```

First run — build the Pinecone knowledge base:

```bash
python ingestion_pc.py
```

Then start the app:

```bash
streamlit run smartMCI.py
```

API 571, 970, and 584 PDFs must be placed in `data/raw_documents/` before running ingestion. The Pinecone indexes created are `api571-damage-mechanisms`, `api970-corrosion-control`, and `api584-integrity-windows`.
