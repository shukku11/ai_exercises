# RAG Plan

## Step 1 - POC Completed So Far
- Built a Python RAG proof of concept using LangChain + Azure OpenAI + Chroma.
- Loaded a PDF from `docs/FY26_US_Incentive_Guide.pdf` and split it into chunks with overlap.
- Created embeddings with Azure embedding deployment and indexed chunks in Chroma.
- Configured Azure chat deployment for answer generation.
- Connected retrieval + generation using a RetrievalQA chain with `chain_type="stuff"`.
- Added an interactive CLI loop to accept repeated user questions until `exit` or `quit`.
- Reorganized project structure under `rag/policy_rag_poc` for clearer ownership.

## Step 2 - Stabilize and Secure
- Move Azure secrets from hardcoded values to environment variables or `.env`.
- Add startup validation for required environment variables and deployment names.
- Add basic logging (question, top retrieved chunks metadata, response time).
- Persist Chroma DB to disk so embeddings are reused between runs.

## Step 3 - Production-Ready Enhancements
- Add source citations in answers (page/chunk references).
- Support ingestion of multiple PDFs from `docs/` with a rebuild/index command.
- Add evaluation script for retrieval quality and answer quality.
- Optionally expose this as a FastAPI endpoint for app integration.
