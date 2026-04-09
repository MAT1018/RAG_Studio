# Investment Analysis RAG System

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline in Python for investment analysis using a PDF textbook.

It includes:

- PDF ingestion with PyMuPDF
- Recursive chunking with LangChain
- Local sentence-transformers embedding generation
- FAISS vector storage with local persistence
- Semantic retrieval (`top_k=3`)
- Strict context-grounded answer generation with a Gemini chat model
- Logging, chunk previews, embedding previews, a CLI workflow, and a Streamlit frontend

## Project Structure

```text
RAG_Assignment/
├── ingest.py
├── chunking.py
├── embed.py
├── retriever.py
├── generator.py
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Gemini API key:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

## How It Works

1. `ingest.py` loads the textbook PDF and extracts page text with PyMuPDF.
2. `chunking.py` splits the text using recursive chunking with:
   - `chunk_size=500`
   - `chunk_overlap=100`
3. `embed.py` creates local sentence-transformers embeddings and stores them in a FAISS index on disk.
4. `retriever.py` performs semantic retrieval with `top_k=3`.
5. `generator.py` builds a strict grounded prompt so the LLM only uses the retrieved context.
6. `main.py` provides a CLI for setup and query execution.
7. `app.py` provides a visual Streamlit frontend for demo-friendly interaction.

## Run the System

### Option 1: Setup (Ingest + Embed)

This step parses the PDF, chunks the text, creates embeddings, and saves the FAISS index locally.

```bash
python main.py setup --pdf "path/to/investment_textbook.pdf" --print-chunks --print-embeddings
```

Optional flags:

- `--persist-dir` to change the FAISS storage path
- `--print-chunks` to print generated chunk previews
- `--print-embeddings` to print sample embedding vectors

### Option 2: Query

This step loads the saved FAISS index, retrieves the most relevant chunks, shows them, and generates a grounded answer.

```bash
python main.py query --question "What are the main differences between value investing and growth investing?"
```

Optional flag:

- `--persist-dir` to point to a different saved FAISS index directory

### Interactive CLI

If you run the app without arguments, it opens a simple menu:

```bash
python main.py
```

Then choose:

- `1` for setup
- `2` for query

## Frontend Demo UI

The project now includes a Streamlit frontend so you can demonstrate the pipeline more visually during your assignment video.

Run it with:

```bash
streamlit run app.py
```

The frontend supports:

- uploading the textbook PDF
- running setup from the browser
- previewing generated chunks
- previewing embedding vector samples
- showing local FAISS index files for backend verification
- one-click use of the five mandatory assignment questions
- asking grounded investment-analysis questions
- viewing retrieved chunks before the final answer

## Assignment Demo Alignment

This project is set up to support the assignment workflow:

1. Upload the provided investment textbook in the Streamlit app.
2. Show the generated chunk previews.
3. Show the embedding previews for at least two chunks.
4. Show the local FAISS persistence files in the "Backend Verification" section.
5. Ask the five mandatory questions exactly as listed in the UI sidebar.
6. Slowly scroll through the retrieved context and final answer during the recording.

Mandatory questions included in the UI:

1. `how to deal with brokerage houses?`
2. `what is theory of diversification?`
3. `how to become intelligent investor?`
4. `how to do business valuation?`
5. `what is putting all eggs in one basket analogy?`

## Output Behavior

The query workflow prints:

1. The retrieved chunks
2. A clean final answer

This makes the full RAG pipeline easy to explain during an academic video demonstration.

## Prompt Design

The answer generation prompt is designed to:

- force strict use of retrieved context only
- reject unsupported answers
- maintain a professional financial expert tone

If the retrieved textbook context is insufficient, the model explicitly says that the context does not contain enough information.

## Notes for Demo

- Run `setup` once for the textbook PDF.
- Use `query` multiple times for different investment questions.
- Turn on `--print-chunks` and `--print-embeddings` during the demo to show the full pipeline visually.
- The FAISS index is persisted locally, so repeated queries are fast after setup.
- Use `streamlit run app.py` if you want a cleaner UI for screen recording.
- Local embeddings avoid API-rate-limit bottlenecks during setup; Gemini is used only for final answer generation.

## Privacy

- Do not commit or share `.env`, uploaded textbook files, or the local FAISS index.
- The included `.gitignore` excludes the virtual environment, local uploads, and vector-store artifacts.
- Use the generated chunks and embeddings only for this coursework submission.
