import streamlit as st
import os
import time
import json
import re
import boto3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import openai
import random
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------
# Load Environment Variables
# ------------------------------
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION_US")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID_PolicyGPT")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_PolicyGPT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BUCKET_NAME = "edme-apps-data-dev"
PREFIX = "policygpt/"
VECTOR_INDEX_ARN = "arn:aws:s3vectors:us-east-1:437639050237:bucket/edmeapps-vector-dev/index/policy-gpt-index-two"

# ------------------------------
# Initialize OpenAI
# ------------------------------
openai.api_key = OPENAI_API_KEY

# ------------------------------
# Initialize AWS clients
# ------------------------------
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

s3vectors = boto3.client(
    "s3vectors",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# ------------------------------
# Helper Functions
# ------------------------------

def extract_text_from_pdf(bucket, s3_key):
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
            resp = textract.get_document_text_detection(JobId=job_id, NextToken=next_token) if next_token else textract.get_document_text_detection(JobId=job_id)
            for block in resp.get("Blocks", []):
                if block.get("BlockType") == "LINE":
                    page_num = block.get("Page", 1)
                    pages.setdefault(page_num, []).append(block.get("Text", ""))
            next_token = resp.get("NextToken")
            if not next_token:
                break

        page_texts = ["\n".join(pages[p]) for p in sorted(pages.keys())]
        return page_texts, None
    except Exception as e:
        return None, str(e)

# def semantic_hierarchical_chunking(page_text, page_num, file_name):
#     """
#     Structure-aware semantic chunking that breaks text into meaningful sections
#     based on semantic boundaries and hierarchical structure
#     """
#     chunks = []
    
#     # Split by common section patterns in insurance documents
#     section_patterns = [
#         r'\n(?=[A-Z][A-Z\s]{10,})\n',  # All caps headers
#         r'\n(?=\d+\.\s+[A-Z])',  # Numbered sections
#         r'\n(?=[A-Z][a-z]+:)',  # Titled sections with colons
#         r'\n(?=SECTION|ARTICLE|CLAUSE|COVERAGE|TERMS|CONDITIONS|DEFINITIONS)',  # Common keywords
#         r'\n\n+',  # Paragraph breaks
#     ]
    
#     # Try splitting by patterns in order of hierarchy
#     sections = [page_text]
#     for pattern in section_patterns:
#         new_sections = []
#         for section in sections:
#             parts = re.split(pattern, section)
#             new_sections.extend([p.strip() for p in parts if p.strip()])
#         sections = new_sections
    
#     # Create semantic chunks with overlap for context preservation
#     chunk_size = 800  # characters
#     overlap = 200  # characters for context continuity
    
#     for section in sections:
#         if len(section) <= chunk_size:
#             chunks.append({
#                 'text': section,
#                 'type': 'section'
#             })
#         else:
#             # Split large sections with overlap
#             start = 0
#             while start < len(section):
#                 end = start + chunk_size
#                 chunk_text = section[start:end]
                
#                 # Try to break at sentence boundary
#                 if end < len(section):
#                     last_period = chunk_text.rfind('.')
#                     last_newline = chunk_text.rfind('\n')
#                     break_point = max(last_period, last_newline)
#                     if break_point > chunk_size * 0.7:  # At least 70% of chunk size
#                         chunk_text = chunk_text[:break_point + 1]
#                         end = start + break_point + 1
                
#                 chunks.append({
#                     'text': chunk_text.strip(),
#                     'type': 'subsection'
#                 })
                
#                 start = end - overlap
    
#     return chunks

def recursive_chunk_with_langchain(page_text, page_num, file_name):
    """
    Use LangChain's RecursiveCharacterTextSplitter for improved chunking
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(page_text)
    
    chunk_records = []
    for idx, chunk in enumerate(chunks):
        chunk_records.append({
            'text': chunk,
            'chunk_index': idx,
            'page_number': page_num,
            'file_name': file_name
        })
    
    return chunk_records

def extract_metadata_llm(page_text, file_name):
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
        data = json.loads(content[start:end+1])
        return {
            "policy_number": data.get("policy_number"),
            "policy_type": data.get("policy_type"),
            "client_name": data.get("client_name"),
            "client_type": data.get("client_type"),
        }
    except Exception as e:
        st.warning(f"Metadata extraction failed for {file_name}: {e}")
        return {"policy_number": None, "policy_type": None, "client_name": None, "client_type": None}

def generate_embedding(text):
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
    try:
        if not vector:
            return False, "No embedding found"
        vector_id = f"{metadata_record['file_name']}::{metadata_record['page_number']}::{int(time.time())}"
        slim_metadata = {
            "file_name": metadata_record["file_name"],
            "page_number": metadata_record["page_number"],
            "policy_number": metadata_record.get("policy_number"),
            "policy_type": metadata_record.get("policy_type"),
            "client_name": metadata_record.get("client_name"),
            "client_type": metadata_record.get("client_type"),
            "timestamp": metadata_record.get("timestamp"),
            "text": metadata_record.get("text")
        }
        s3vectors.put_vectors(
            indexArn=VECTOR_INDEX_ARN,
            vectors=[{"key": vector_id, "data": {"float32": vector}, "metadata": slim_metadata}]
        )
        return True, vector_id
    except Exception as e:
        return False, str(e)

# ------------------------------
# SESSION-BASED RAG Helper Functions
# ------------------------------
def get_query_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def retrieve_documents_from_session(query, session_chunks, top_k=10):
    """
    Retrieve documents from current session only using cosine similarity
    """
    query_vector = get_query_embedding(query)
    
    # Calculate cosine similarity for each chunk
    results = []
    for chunk in session_chunks:
        if chunk.get('vector'):
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(query_vector, chunk['vector']))
            query_norm = sum(a * a for a in query_vector) ** 0.5
            chunk_norm = sum(a * a for a in chunk['vector']) ** 0.5
            similarity = dot_product / (query_norm * chunk_norm)
            
            results.append({
                'chunk': chunk,
                'similarity': similarity,
                'text': chunk['text'],
                'metadata': {
                    'file_name': chunk['file_name'],
                    'page_number': chunk['page_number'],
                    'chunk_index': chunk.get('chunk_index', 0)
                }
            })
    
    # Sort by similarity (higher is better)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results[:top_k]

def retrieve_documents(query, top_k=10):
    query_vector = get_query_embedding(query)
    response = s3vectors.query_vectors(
        indexArn=VECTOR_INDEX_ARN,
        queryVector={"float32": query_vector},
        topK=top_k,
        returnDistance=True,
        returnMetadata=True
    )
    results = response.get("vectors", [])
    docs = []
    for r in results:
        meta = r.get("metadata", {})
        text_snippet = meta.get("text", "")
        docs.append({
            "key": r.get("key"),
            "distance": r.get("distance"),
            "text": text_snippet,
            "metadata": meta
        })
    return docs

def extract_numbers_from_docs(docs):
    numbers = []
    for doc in docs:
        # Extract all numbers including decimals and currency formats
        text = doc.get('text', '')
        raw_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
        # Convert to float and clean commas
        for num_str in raw_numbers:
            try:
                clean_num = float(num_str.replace(',', ''))
                if clean_num > 0:
                    numbers.append(clean_num)
            except:
                pass
    return numbers

def generate_answer_from_session(query, docs, numbers, uploaded_files):
    """
    Generate answer using only current session documents
    """
    # Build detailed context with citations
    context_parts = []
    for idx, d in enumerate(docs, 1):
        metadata = d.get('metadata', {})
        page_num = metadata.get('page_number', 'Unknown')
        file_name = metadata.get('file_name', 'Unknown')
        similarity = d.get('similarity', 'N/A')
        
        context_parts.append(f"""
[Citation {idx}] - Page {page_num} of {file_name} (Similarity Score: {similarity})
Content: {d['text'][:1000]}
""")
    
    context = "\n".join(context_parts)
    numbers_summary = f"All extracted numbers from documents: {numbers}" if numbers else "No numerical values found."
    
    # Add session file information
    session_files_info = f"\n\nCURRENT SESSION UPLOADED FILES: {', '.join(uploaded_files)}"
    
    prompt = f"""
You are a helpful and accurate policy assistant. Use ONLY the retrieved documents from the CURRENT SESSION below to answer the user's query.

CRITICAL INSTRUCTIONS:
1. You can ONLY reference documents uploaded in the CURRENT SESSION: {', '.join(uploaded_files)}
2. If the user asks "Which file did I upload?" or similar questions, list ONLY these files: {', '.join(uploaded_files)}
3. Provide accurate information based ONLY on the documents provided below
4. Include ALL relevant numbers, amounts, dates, and percentages from the documents
5. When mentioning information, ALWAYS cite the source using [Citation X] format
6. If multiple citations support the same point, list all of them like [Citation 1, 2]
7. Do not make up or assume any information not present in the documents
8. If the answer is not in the documents, clearly state that
9. NEVER reference any documents not in the current session list above

Retrieved Documents from Current Session:
{context}

{numbers_summary}
{session_files_info}

User Query: {query}

Answer (with citations):"""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a knowledgeable policy assistant that answers questions based ONLY on documents from the current session. Current session files: {', '.join(uploaded_files)}. Never reference documents outside this list."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

def generate_answer(query, docs, numbers):
    # Build detailed context with citations
    context_parts = []
    for idx, d in enumerate(docs, 1):
        page_num = d['metadata'].get('page_number', 'Unknown')
        file_name = d['metadata'].get('file_name', 'Unknown')
        distance = d.get('distance', 'N/A')
        
        context_parts.append(f"""
[Citation {idx}] - Page {page_num} of {file_name} (Relevance Score: {distance})
Content: {d['text'][:1000]}
""")
    
    context = "\n".join(context_parts)
    numbers_summary = f"All extracted numbers from documents: {numbers}" if numbers else "No numerical values found."
    
    prompt = f"""
You are a helpful and accurate policy assistant. Use the retrieved documents below to answer the user's query.

IMPORTANT INSTRUCTIONS:
1. Provide accurate information based ONLY on the documents provided
2. Include ALL relevant numbers, amounts, dates, and percentages from the documents
3. When mentioning information, ALWAYS cite the source using [Citation X] format
4. If multiple citations support the same point, list all of them like [Citation 1, 2]
5. Do not make up or assume any information not present in the documents
6. If the answer is not in the documents, clearly state that

Retrieved Documents:
{context}

{numbers_summary}

User Query: {query}

Answer (with citations):"""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a knowledgeable policy assistant that always provides accurate, well-cited answers based on source documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Policy GPT", layout="wide")
st.title("📘 Policy GPT — Upload PDF & Query")

# Sidebar
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file to upload", type=["pdf"])

# Initialize session state
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "session_chunks" not in st.session_state:
    st.session_state.session_chunks = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ------------------------------
# Upload PDF and process (ONLY if new file uploaded)
# ------------------------------
if uploaded_file and (st.session_state.processed_file != uploaded_file.name):
    file_name = uploaded_file.name
    with st.spinner("📤 Uploading and processing PDF..."):
        s3_key = f"{PREFIX}{file_name}"
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
        all_chunks = []
        
        for idx, page_text in enumerate(page_texts, start=1):
            st.info(f"Processing Page {idx} with LangChain Recursive Chunking...")
            
            # Use LangChain recursive chunking
            chunks = recursive_chunk_with_langchain(page_text, idx, file_name)
            
            # Extract metadata from the page
            metadata = extract_metadata_llm(page_text, file_name)
            
            # Process each chunk
            for chunk in chunks:
                chunk_text = chunk['text']
                
                # Generate embedding for chunk
                vector = generate_embedding(chunk_text)
                
                # Create metadata record for chunk
                metadata_record = {
                    "page_number": idx,
                    "chunk_index": chunk['chunk_index'],
                    "sequenceno": idx,
                    "file_name": file_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "text": chunk_text,
                    "vector": vector,
                    **metadata
                }
                
                # Store in S3 Vector Index
                store_vector_with_metadata(vector, metadata_record)
                
                # Store in session chunks for current session retrieval
                all_chunks.append(metadata_record)
                records.append(metadata_record)
        
        # Update session state
        st.session_state.session_chunks.extend(all_chunks)
        st.session_state.metadata = records
        st.session_state.processed_file = file_name
        
        # Add to uploaded files list
        if file_name not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(file_name)
        
        st.success(f"✅ PDF processed with LangChain recursive chunking! Total chunks: {len(all_chunks)}")

# ------------------------------
# Display chat history FIRST
# ------------------------------
st.subheader("💬 Chat with Policy GPT")

# Display uploaded files info
if st.session_state.uploaded_files:
    st.info(f"📄 Current session files: {', '.join(st.session_state.uploaded_files)}")

for chat in st.session_state.chat_history:
    # User query box
    with st.container():
        st.markdown("**🧑 You:**")
        st.info(chat['user'])
    
    # Bot reply box
    with st.container():
        st.markdown("**🤖 Policy GPT:**")
        st.success(chat['bot'])
    
    st.divider()

# ------------------------------
# Query user input
# ------------------------------
query = st.chat_input("Ask a question about the uploaded PDF:")

if query:
    with st.spinner("🤔 Thinking..."):
        # Use session-based retrieval
        docs = retrieve_documents_from_session(query, st.session_state.session_chunks, top_k=10)
        numbers = extract_numbers_from_docs(docs)
        answer = generate_answer_from_session(query, docs, numbers, st.session_state.uploaded_files)
    
    # Add to chat history
    st.session_state.chat_history.append({"user": query, "bot": answer})
    
    # Rerun to display the new message
    st.rerun()

# Display metadata table (hidden when chat history exists)
if st.session_state.metadata and len(st.session_state.chat_history) == 0:
    st.subheader("🧾 Extracted Metadata Records")
    df = pd.DataFrame(st.session_state.metadata)
    display_columns = [
        "sequenceno", "page_number", "chunk_index", "file_name",
        "policy_number", "policy_type", "client_type",
        "client_name", "timestamp", "vector","text"
    ]
    df = df[[c for c in display_columns if c in df.columns]]
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download Metadata CSV",
        df.to_csv(index=False),
        file_name="metadata_records.csv",
        mime="text/csv"
    )
