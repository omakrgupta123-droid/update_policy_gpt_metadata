import streamlit as st
import boto3
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import openai

# -------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION_US")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID_PolicyGPT")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_PolicyGPT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# -------------------------------------------------------------
# AWS Clients
# -------------------------------------------------------------
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

textract = boto3.client(
    "textract",
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

s3vector = boto3.client(
    "s3vectors",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
BUCKET_NAME = "edme-apps-data-dev"
PREFIX = "policygpt/"
VECTOR_INDEX_ARN = "arn:aws:s3vectors:us-east-1:437639050237:bucket/edmeapps-vector-dev/index/policy-gpt-index-two"

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def extract_text_from_pdf(bucket, s3_key):
    """Extract text from PDF using AWS Textract."""
    try:
        response = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": s3_key}}
        )
        job_id = response["JobId"]

        status = "IN_PROGRESS"
        while status == "IN_PROGRESS":
            time.sleep(5)
            response = textract.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]

        if status != "SUCCEEDED":
            return None, f"Textract failed with status {status}"

        pages = {}
        next_token = None
        while True:
            if next_token:
                response = textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
            else:
                response = textract.get_document_text_detection(JobId=job_id)

            for block in response.get("Blocks", []):
                if block.get("BlockType") == "LINE":
                    page_num = block.get("Page", 1)
                    pages.setdefault(page_num, []).append(block.get("Text", ""))

            next_token = response.get("NextToken")
            if not next_token:
                break

        page_texts = ["\n".join(pages[p]) for p in sorted(pages.keys())]
        return page_texts, None

    except Exception as e:
        return None, str(e)


def extract_metadata_llm(page_text, file_name):
    """Extract metadata from text using GPT-4o."""
    prompt = f"""
You are an expert data extraction model.
Extract structured insurance policy metadata from the text below.
Return valid JSON with all fields even if null.

Text:
{page_text}

Output JSON:
{{
    "policy_number": "<string or null>",
    "policy_type": "<string or null>",
    "client_name": "<string or null>",
    "client_type": "<string or null>"
}}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )
        content = response.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON returned from LLM")
        json_str = content[start:end+1]
        data = json.loads(json_str)
        return {
            "policy_number": data.get("policy_number"),
            "policy_type": data.get("policy_type"),
            "client_name": data.get("client_name"),
            "client_type": data.get("client_type"),
        }
    except Exception as e:
        st.warning(f"Metadata extraction failed for {file_name}: {e}")
        return {
            "policy_number": None,
            "policy_type": None,
            "client_name": None,
            "client_type": None
        }


def generate_embedding(text):
    """Generate embeddings using OpenAI API."""
    try:
        emb = openai.embeddings.create(
            model="text-embedding-3-large",
            input=text[:8000]
        )
        return emb.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding failed: {e}")
        return None


def store_vector_with_metadata(vector, metadata_record):
    """Store embedding vector + metadata in AWS S3 Vector Index."""
    try:
        if not vector:
            return False, "No embedding found"

        vector_id = f"{metadata_record['file_name']}::{metadata_record['page_number']}::{int(time.time())}"

        s3vector.put_vectors(
            indexArn=VECTOR_INDEX_ARN,
            vectors=[
                {
                    "key": vector_id,
                    "data": {"float32": vector},
                    "metadata": metadata_record
                }
            ]
        )
        return True, vector_id
    except Exception as e:
        return False, str(e)


# -------------------------------------------------------------
# RAG - Vector Search
# -------------------------------------------------------------
def _cosine_similarity_matrix(emb_matrix, query_vec):
    """Compute cosine similarity between matrix (n x d) and single vector (d)."""
    emb_norms = np.linalg.norm(emb_matrix, axis=1)
    q_norm = np.linalg.norm(query_vec)
    emb_norms[emb_norms == 0] = 1e-12
    if q_norm == 0:
        q_norm = 1e-12
    sims = (emb_matrix @ query_vec) / (emb_norms * q_norm)
    return sims


def retrieve_relevant_chunks(query, top_k=3):
    q_emb = generate_embedding(query)
    if not q_emb:
        return []

    try:
        if hasattr(s3vector, "query"):
            resp = s3vector.query(
                indexArn=VECTOR_INDEX_ARN,
                queryVector={"float32": q_emb},
                topK=top_k,
                includeMetadata=True
            )
            matches = resp.get("matches", [])
            normalized = []
            for m in matches:
                meta = m.get("metadata", {})
                score = m.get("score", None) or m.get("distance", None) or 0.0
                normalized.append({"metadata": meta, "score": score})
            return normalized
    except Exception as e:
        st.warning(f"Server-side vector query failed or unavailable, falling back to local retrieval: {e}")

    local_records = st.session_state.get("metadata", []) or []
    vectors = []
    metadatas = []
    for rec in local_records:
        vec = rec.get("vector")
        if vec and isinstance(vec, (list, tuple)) and len(vec) > 0:
            vectors.append(np.array(vec, dtype=float))
            metadatas.append(rec)

    if not vectors:
        return []

    emb_matrix = np.vstack(vectors)
    q_vec = np.array(q_emb, dtype=float)
    sims = _cosine_similarity_matrix(emb_matrix, q_vec)
    top_k = min(top_k, len(sims))
    top_idx = np.argsort(-sims)[:top_k]

    matches = []
    for idx in top_idx:
        matches.append({"metadata": metadatas[int(idx)], "score": float(sims[int(idx)])})
    return matches


# -------------------------------------------------------------
# DYNAMIC RAG RESPONSE LOGIC
# -------------------------------------------------------------
def generate_rag_response(model_name, query):
    """Generate context-aware, dynamically detailed RAG response."""
    matches = retrieve_relevant_chunks(query, top_k=4)

    if not matches:
        context_text = "No relevant context found in the vector database."
    else:
        pieces = []
        for m in matches:
            md = m["metadata"]
            page_info = f"(file: {md.get('file_name')}, page: {md.get('page_number')}, score: {m.get('score'):.4f})"
            text_snippet = md.get("text", "")
            pieces.append(f"{page_info}\n{text_snippet}")
        combined_context = "\n\n---\n\n".join(pieces)

    # Detect if user wants detailed or summary output
    query_lower = query.lower()
    if any(word in query_lower for word in ["detail", "summary", "explain fully", "comprehensive"]):
        detail_level = "detailed"
    else:
        detail_level = "concise"

    if detail_level == "detailed":
        prompt = f"""
You are a professional insurance document analyst.

Analyze the provided PDF context and answer the user’s question clearly and accurately.
Identify the policyholder, policy number, and key policy details.

------
**Question:** {query}
------
**Context Extracted from PDFs:**
{combined_context}
------

**Response Format:**
1. Begin with a one-line summary (e.g., “The policy for [Client Name] (Policy No. …) covers the following risks.”)
2. Present organized details by state or region using icons:
   - 🏭 Industrial zones  
   - ⚓ Ports  
   - ⚙️ Manufacturing / production  
   - 🛢️ Tank / chemical terminals  
3. Use clean Markdown bullet points — one location per line.
4. Add a **💡 Summary** section:
   - Total risk sites  
   - States covered  
   - Nature of risk (if mentioned)  
   - Sum insured (if available)
5. End with a brief **Conclusion** (1–2 lines).

Be concise, factual, and neatly formatted.
"""
    else:
        prompt = f"""
You are a professional insurance assistant.
Provide a short, direct answer (2–3 lines maximum) to the user’s question
based only on the context below.

Question: {query}

Context:
{combined_context}
"""

    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800 if detail_level == "detailed" else 300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"RAG response failed: {e}"


# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="PolicyGPT RAG System", layout="wide")
st.title("📘 EDME PolicyGPT")

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Select LLM Model", ["gpt-4", "gpt-4o"])
    st.markdown("---")
    st.caption("This app processes PDFs, extracts metadata & embeddings, and allows RAG-based querying.")

if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

left, right = st.columns([1, 1])

with left:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        file_name = uploaded_file.name
        s3_key = f"{PREFIX}{file_name}"

        with st.spinner("📤 Uploading and processing PDF..."):
            try:
                s3.upload_fileobj(uploaded_file, BUCKET_NAME, s3_key)
                st.success("✅ File uploaded to S3 successfully!")
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

            page_texts, error = extract_text_from_pdf(BUCKET_NAME, s3_key)
            if error:
                st.error(error)
                st.stop()

            records = []
            for idx, page_text in enumerate(page_texts, start=1):
                st.info(f"Processing Page {idx}...")
                metadata = extract_metadata_llm(page_text, file_name)

                metadata_record = {
                    "page_number": idx,
                    "sequenceno": idx,
                    "file_name": file_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "text": page_text,
                    **metadata
                }

                vector = generate_embedding(page_text)
                metadata_record["vector"] = vector

                success, msg = store_vector_with_metadata(vector, metadata_record)
                metadata_record["store_status"] = msg if success else f"Error: {msg}"

                records.append(metadata_record)

            st.session_state.metadata = records
            st.success("✅ PDF processed, embeddings stored — ready for RAG chat!")

            st.info(f"📄 Processed {len(records)} pages and stored {len([r for r in records if r.get('vector')])} vectors.")

    if st.session_state.metadata:
        df = pd.DataFrame(st.session_state.metadata)
        display_columns = [
            "vector", "text", "sequenceno", "page_number",
            "file_name", "policy_number", "policy_type",
            "client_type", "client_name", "timestamp"
        ]
        display_columns = [c for c in display_columns if c in df.columns]
        df = df[display_columns]
        st.subheader("🧾 Extracted and Stored Metadata Records")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download Metadata CSV",
            df.to_csv(index=False),
            file_name="metadata_records.csv",
            mime="text/csv"
        )

with right:
    st.header("")
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

st.markdown("---")
st.header("💬 Ask questions about the uploaded policy (RAG)")

user_input = st.chat_input("Ask something about your uploaded policies...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rag_response = generate_rag_response(model_choice, user_input)
            st.markdown(rag_response)

    st.session_state.chat_history.append({"role": "assistant", "content": rag_response})


