import os
import re
from typing import List, TypedDict
from dotenv import load_dotenv

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

load_dotenv()

# --- 1. Configuration & Vector Store ---
# pinecone_api_key = os.getenv("pinecone_api_key")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

pinecone_api_key = st.secrets["pinecone_api_key"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


embedding_mod = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(
    index_name="medical-summarize",
    embedding=embedding_mod,
    pinecone_api_key=pinecone_api_key
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 2. LLM Definition ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, # Low temperature for medical accuracy
    api_key=GEMINI_API_KEY
)

# --- 3. Graph State Definition ---
class GraphState(TypedDict):
    """Represents the state of our agent."""
    question: str
    generation: str
    documents: List[Document]

# --- 4. Node Functions ---

def retrieve(state: GraphState):
    """Retrieve documents from Pinecone."""
    print("---RETRIEVING EXAMPLES---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """Generate summary using retrieved examples as guidance."""
    print("---GENERATING SUMMARY---")
    question = state["question"]
    documents = state["documents"]
    
    # Constructing the Few-Shot context from Pinecone metadata
    # Assuming your Pinecone docs have 'text' and 'summary' in metadata
    context_examples = ""
    for i, doc in enumerate(documents):
        # Accessing metadata fields specifically
        txt = doc.metadata.get('text', 'N/A')
        summ = doc.metadata.get('summary', 'N/A')
        context_examples += f"\nExample {i+1}:\nOriginal Text: {txt}\nTarget Summary: {summ}\n"

    template = """You are an expert Medical Scribe. Your task is to summarize the user's medical report text.
    
    Below are examples of how similar medical texts have been summarized in the past. Use these as a style and structure guide:
    {examples}
    
    User's New Medical Report Text:
    {input}
    
    Instructions:
    1. Analyze the 'Target Summary' style from the examples.
    2. Extract key medical findings, diagnosis, and recommendations from the user's text.
    3. Provide a concise summary eg "We present a case of 20-year-old male with an isolated scaphoid dislocation and scapholunate ligament injury of the wrist, diagnosed and repaired in an acute setting with k wires and suture anchor augmentation. At 1 year follow up patient complained of no pain and returned to work without any limitations, with no signs of avascular necrosis of the scaphoid on imaging.".
    
    Final Summary:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm
    
    response = rag_chain.invoke({"examples": context_examples, "input": question})
    return {"generation": response.content, "question": question}

# --- 5. Build the LangGraph ---

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Build the edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the app
app = workflow.compile()

# --- 6. Streamlit Integration ---
import io
from typing import Optional

# UI constants
MAX_UPLOAD_MB = 2  # Max upload size in megabytes
MAX_CHARS = 80000  # Max characters to send to the LLM (truncation fallback)


def generate_agentic_response(user_input: str) -> str:
    """Runs the LangGraph agent."""
    inputs = {"question": user_input}
    output = app.invoke(inputs)

    # Cleaning response for a clean UI output
    final_text = output["generation"]
    cleaned_response = re.sub(r"^\s*[-–—]+\s*", "", final_text)
    return cleaned_response.strip()


def extract_text_from_docx(file_stream: io.BytesIO) -> str:
    try:
        from docx import Document as DocxDocument
    except Exception as e:
        st.error("To extract .docx files please install `python-docx` (pip install python-docx).")
        raise

    doc = DocxDocument(file_stream)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    # Prefer pdfplumber for better text extraction
    try:
        import pdfplumber
        with pdfplumber.open(file_stream) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
    except Exception:
        # Fallback to PyPDF2
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file_stream)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception:
            st.error("To extract PDFs, please install `pdfplumber` or `PyPDF2` (pip install pdfplumber PyPDF2).")
            raise


def extract_text_from_doc(file_stream: io.BytesIO) -> str:
    # .doc (old Word) support is best-effort; recommend converting to .docx
    try:
        import textract
        text = textract.process(file_stream)
        return text.decode("utf-8", errors="ignore")
    except Exception:
        st.warning(".doc files are not reliably supported. Please convert to .docx or paste the text directly.")
        return ""


def extract_text_from_uploaded(uploaded_file) -> Optional[str]:
    if not uploaded_file:
        return None

    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Uploaded file is too large (> {MAX_UPLOAD_MB} MB). Please upload a smaller file or paste the text directly.")
        return None

    file_bytes = uploaded_file.read()
    ext = uploaded_file.name.lower().split(".")[-1]
    file_stream = io.BytesIO(file_bytes)

    if ext in ("txt", "text"):
        try:
            return file_bytes.decode("utf-8")
        except Exception:
            return file_bytes.decode("latin-1")
    elif ext == "docx":
        return extract_text_from_docx(file_stream)
    elif ext == "pdf":
        return extract_text_from_pdf(file_stream)
    elif ext == "doc":
        return extract_text_from_doc(file_stream)
    else:
        st.error("Unsupported file type. Supported: .txt, .docx, .pdf, (best-effort .doc).")
        return None


# Chunking and multi-pass summarization settings
CHUNK_SIZE = 40000  # chars per chunk for multi-pass
CHUNK_OVERLAP = 500  # overlap between chunks to preserve context
HARD_CHAR_CAP = 300000  # absolute max characters to process (will truncate beyond this)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def multi_pass_summarize(text: str) -> str:
    """Summarize large text by chunking and iteratively reducing summaries.

    Steps:
    1) Chunk the current text and summarize each chunk.
    2) Join chunk summaries and repeat until the combined summary fits within MAX_CHARS
       or a max number of passes is reached.
    """
    current = text
    max_passes = 5
    pass_num = 0

    while True:
        # If the current text is small enough for a single-pass summary, do it.
        if len(current) <= MAX_CHARS:
            return generate_agentic_response(current)

        if pass_num >= max_passes:
            # Force a final summarization by truncating to MAX_CHARS
            truncated = current[:MAX_CHARS]
            return generate_agentic_response(truncated)

        # Chunk and summarize each piece
        chunks = chunk_text(current)
        summaries: List[str] = []
        progress = st.progress(0)
        status = st.empty()
        for i, chunk in enumerate(chunks):
            status.text(f"Summarizing chunk {i+1}/{len(chunks)} (pass {pass_num+1})...")
            try:
                s = generate_agentic_response(chunk)
            except Exception as e:
                s = f"[Error summarizing chunk {i+1}: {e}]"
            summaries.append(s)
            progress.progress(int((i+1) / len(chunks) * 100))

        # Prepare for the next pass
        current = "\n\n".join(summaries)
        pass_num += 1
        progress.empty()
        status.empty()


# Streamlit UI
if __name__ == "__main__":
    st.set_page_config(page_title="Medical Report Summarizer", layout="wide")

    st.markdown("""
    <style>
        /* Force light color scheme */
        :root { color-scheme: light; }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .block-container, .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        [data-testid="stSidebar"], .css-1d391kg, .css-185krt0 {
            background-color: #f8f9fa !important;
            color: #000000 !important;
        }
        header, footer, [data-testid="stToolbar"] { background-color: transparent !important; color: #000000 !important; }
        /* App-specific styles */
        .big-title {font-size:32px; font-weight:700; color:#000000 !important;}
        .muted {color:#6c757d !important;}
        .card {background-color:#f8f9fa; padding:16px; border-radius:8px; color:#000000 !important;}
        /* Prevent browser 'force dark mode' from inverting images/icons */
        img, svg { filter: none !important; }
        /* Ensure inputs and text areas are dark text on light background */
        textarea, input, .stTextInput, .stTextArea { background-color: #ffffff !important; color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown('<div class="big-title">✅ Medical Report Summarizer (Agentic-RAG)</div>', unsafe_allow_html=True)
        st.markdown("<div class='muted'>Upload a medical report (.docx/.pdf/.txt) or paste the text below. Max upload: 2MB. Large documents will be processed using automatic chunking and multi-pass summarization.</div>", unsafe_allow_html=True)
    with header_col2:
        st.markdown("<div style='text-align:right'><small class='muted'>v1.1 • Enhanced UI</small></div>", unsafe_allow_html=True)

    st.write("---")

    left, right = st.columns([2, 3])

    # Left: Upload / Input controls
    with left:
        st.header("Input")
        uploaded_file = st.file_uploader("Upload report file (Max 2MB)", type=["pdf", "docx", "doc", "txt"], help="Supports .docx, .pdf, .txt. .doc is best-effort. Max 2MB per file.")
        st.caption("Limit 2MB per file • PDF, DOCX, DOC, TXT")
        pasted_text = st.text_area("Or paste the report text here:", height=200)

        # Note: Advanced options removed — multi-pass chunking is applied automatically for large documents.
        submit = st.button("Generate Summary", type="primary")

    # Right: Preview and output
    with right:
        st.header("Preview & Output")
        preview_card = st.container()

        # Get text from either uploaded file or pasted text
        source_text = None
        source_name = None
        if uploaded_file is not None:
            try:
                source_text = extract_text_from_uploaded(uploaded_file)
                source_name = uploaded_file.name
            except Exception:
                source_text = None
        if not source_text and pasted_text:
            source_text = pasted_text
            source_name = "Pasted Text"

        if source_text:
            char_count = len(source_text)
            words = len(source_text.split())
            preview_card.markdown(f"**Source:** {source_name or 'Unknown'}  \n**Characters:** {char_count:,}  \n**Words:** {words:,}")

            if char_count > HARD_CHAR_CAP:
                st.warning(f"Source text ({char_count:,} chars) exceeds the hard cap ({HARD_CHAR_CAP:,}). The text will be truncated to the first {HARD_CHAR_CAP:,} characters.")
                source_text = source_text[:HARD_CHAR_CAP]
                char_count = len(source_text)
                words = len(source_text.split())

            if char_count > MAX_CHARS:
                chunks = chunk_text(source_text)
                st.info(f"Large document detected — will be processed in {len(chunks)} chunks (≈{CHUNK_SIZE:,} chars each) using multi-pass summarization. This may take longer.")

            with st.expander("Preview source text", expanded=False):
                st.write(source_text[:10000] + ("\n\n..." if len(source_text) > 10000 else ""))

        else:
            st.info("No text loaded yet. Upload a file or paste the report text to begin.")

        # Display result area
        result_area = st.container()

        if submit:
            if not source_text:
                st.warning("Please upload a supported file or paste some text before generating a summary.")
            else:
                with st.spinner("Generating professional summary..."):
                    try:
                        if len(source_text) <= MAX_CHARS:
                            result = generate_agentic_response(source_text)
                        else:
                            result = multi_pass_summarize(source_text)

                        result_area.subheader("Professional Summary")
                        st.markdown(f"<div class='card'><pre style='white-space:pre-wrap'>{result}</pre></div>", unsafe_allow_html=True)

                        st.download_button("Download summary as text", data=result, file_name="summary.txt")
                    except Exception as e:
                        st.error(f"An error occurred while generating the summary: {e}")

    # Helpful footer
    st.write("---")
    st.markdown("**Tips:** Use short, focused reports for best results. For very long reports, the app will automatically process them in chunks, but you may get more targeted results by splitting into logical sections and summarizing each separately.")
