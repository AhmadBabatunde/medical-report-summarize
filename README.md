# Medical Report Summarizer (Agentic-RAG)

A Streamlit app that summarizes medical reports using a retrieval-augmented generative agent (LangGraph + Gemini). It supports uploading `.docx`, `.pdf`, `.txt`, and best-effort `.doc` files as well as pasted text, and includes automatic chunking + multi-pass summarization for large documents.

---

## Features ✅

- Upload `.docx`, `.pdf`, `.txt` (and `.doc` best-effort) or paste text directly
- File upload limit: **2 MB** per file (enforced)
- Automatic chunking + multi-pass summarization for long documents (no manual options required)
- Uses Pinecone retriever for few-shot examples and Gemini (ChatGoogleGenerativeAI) for generation
- Download summary as a `.txt` file

---

## Quick start 🚀

1. Create and activate a virtual environment (recommended):

   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Set environment variables (add to a `.env` file or your environment):

   PINECONE_API_KEY=your_pinecone_key
   GEMINI_API_KEY=your_gemini_api_key

4. Run the app:

   streamlit run app.py

Then open the URL shown by Streamlit in your browser.

---

## Notes & Troubleshooting 🔧

- File extraction relies on optional libraries:
  - `.docx`: `python-docx`
  - `.pdf`: `pdfplumber` (preferred) or `PyPDF2` (fallback)
  - `.doc`: `textract` (best-effort; may require system packages)
- If you see extraction errors, install the packages above or paste the text directly.
- If a report is extremely long (>300,000 characters), the app truncates it to a safe hard cap to avoid exceeding the LLM context.
- To change chunking/limits, edit these constants in `app.py`:
  - `MAX_UPLOAD_MB`, `MAX_CHARS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `HARD_CHAR_CAP`

---

## Development tips 🛠️

- To pin exact versions, run `pip freeze > requirements.txt` after installing and testing.
- Use the `Hide uploader helper note (experimental)` checkbox in the UI to suppress the platform helper text if it appears.

---

## License

MIT-style. Adjust as needed for your project.

---

If you'd like, I can also add a CONTRIBUTING.md, unit tests for the extractor functions, or a sample test dataset. Which would you prefer next?