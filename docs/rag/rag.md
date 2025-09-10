# Retrieval-Augmented Generation (RAG)

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines **large language models (LLMs)** with **external knowledge retrieval**.
Instead of relying only on pretraining, RAG integrates **dynamic document fetching** to provide more accurate, up-to-date, and grounded answers.

---

## Process

1. **User Query**
   → Example: "What’s the revenue of Apple in 2024?"
1. **Retriever**
   - Searches knowledge bases (vector DBs, search engines, APIs).
   - Returns relevant documents or passages.
1. **LLM (Generator)**
   - Reads the retrieved documents.
   - Synthesizes an answer grounded in the evidence.

---

## Advantages

- **Up-to-date information** (unlike static pretrained models).
- **Domain-specific adaptation** using private data.
- **Explainable responses** with source citations.

---

## Common Tools in RAG

- **Vector Databases**: Pinecone, Weaviate, Milvus, FAISS.
- **Embedding Models**: Convert text into dense vectors for retrieval.
- **Frameworks**: LangChain, LlamaIndex, Haystack.

---

## Example Applications

- Enterprise search assistants.
- Legal, medical, or financial research bots.
- Customer support over proprietary document bases.
- Scientific paper summarization with citations.

---

## Visualization

```
User Query ──► Retriever (Vector DB, Search) ──► Context ──► LLM ──► Grounded Answer
```
