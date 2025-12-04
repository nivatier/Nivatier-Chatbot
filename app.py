# ========================= app.py =========================
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import faiss
import openai
import tiktoken
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ========================= CONFIG =========================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
VECTOR_DB_DIR = "vector_db"
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
SHEET_NAME = "Sheet1"
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")

if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set")
if not GOOGLE_CREDS_JSON:
    raise RuntimeError("GOOGLE_CREDS_JSON not set")

try:
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    sheets_service = build("sheets", "v4", credentials=creds)
    sheet = sheets_service.spreadsheets()
except Exception as e:
    raise RuntimeError(f"Failed to load Google credentials: {e}")

# ========================= FASTAPI INIT =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================= GLOBAL STATE =========================
faiss_index = None
combined_chunks: List[Dict] = []
index_dimension = None
sequential_indexes: List[faiss.IndexFlatL2] = []
use_sequential_search = False

# ========================= Pydantic Models =========================
class ChatMessage(BaseModel):
    user: str
    bot: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    clientId: str

class ChatResponse(BaseModel):
    reply: str

# ========================= UTILITIES =========================
def log_to_google_sheet(row_data: List):
    try:
        sheet.values().append(
            spreadsheetId=SHEET_ID,
            range=f"{SHEET_NAME}!A1",
            valueInputOption="USER_ENTERED",
            body={"values": [row_data]}
        ).execute()
    except Exception as e:
        print("Failed to log to Google Sheets:", e)

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_openai_embedding(text: str) -> Tuple[List[float], int]:
    """
    Returns embedding + tokens used.
    """
    resp = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return resp['data'][0]['embedding'], resp['usage']['total_tokens']

# ========================= FAISS LOAD =========================
def load_vector_dbs() -> bool:
    """
    Loads all FAISS indexes + lightweight JSON chunks.
    Merges indexes if possible. Dimension detection from FAISS.
    """
    global faiss_index, combined_chunks, index_dimension, sequential_indexes, use_sequential_search

    if not os.path.exists(VECTOR_DB_DIR):
        print(f"Vector DB directory '{VECTOR_DB_DIR}' missing")
        return False

    index_files = [f for f in os.listdir(VECTOR_DB_DIR) if f.endswith(".index")]
    if not index_files:
        print("No FAISS indexes found")
        return False

    combined_chunks = []
    indexes = []
    dims = set()

    for idx_file in sorted(index_files):
        base = idx_file[:-len(".index")]
        idx_path = os.path.join(VECTOR_DB_DIR, idx_file)
        meta_path = os.path.join(VECTOR_DB_DIR, f"{base}_metadata.json")

        try:
            index = faiss.read_index(idx_path)
            dim = index.d
            dims.add(dim)
            indexes.append(index)
        except Exception as e:
            print(f"Failed to load index {idx_path}: {e}")
            continue

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    combined_chunks.extend(metadata.get("chunks", []))
            except Exception as e:
                print(f"Failed to read metadata {meta_path}: {e}")
        else:
            print(f"Metadata {meta_path} missing (continuing)")

    if len(dims) != 1:
        print("Incompatible index dimensions:", dims)
        return False

    index_dimension = dims.pop()

    # Merge indexes
    if len(indexes) == 1:
        faiss_index = indexes[0]
    else:
        try:
            faiss_index = faiss.concat_indexes(indexes)
        except Exception:
            print("Failed to merge indexes, enabling sequential search fallback")
            sequential_indexes = indexes
            faiss_index = None
            use_sequential_search = True

    print(f"✅ Loaded {len(combined_chunks)} chunks from {len(indexes)} indexes. Dimension: {index_dimension}")
    return True

# ========================= SEARCH =========================
def search_chunks(query: str, k: int = 5) -> Tuple[List[Dict], int]:
    """
    Returns top-k chunks and embedding token usage.
    """
    global faiss_index, combined_chunks, index_dimension, sequential_indexes, use_sequential_search

    q_emb, embed_tokens = get_openai_embedding(query)
    q_vec = np.array(q_emb, dtype="float32").reshape(1, -1)

    results = []

    if faiss_index is not None:
        D, I = faiss_index.search(q_vec, k)
        for idx in I[0]:
            if idx < len(combined_chunks):
                results.append(combined_chunks[idx])
        return results, embed_tokens

    if use_sequential_search and sequential_indexes:
        all_scores, all_idxs = [], []
        base_offset = 0
        for idx in sequential_indexes:
            try:
                D, I = idx.search(q_vec, k)
                for dist, local_idx in zip(D[0], I[0]):
                    if local_idx == -1:
                        continue
                    global_idx = base_offset + local_idx
                    all_scores.append(float(dist))
                    all_idxs.append(global_idx)
            except Exception:
                continue
            base_offset += idx.ntotal

        ranked = sorted(zip(all_scores, all_idxs), key=lambda x: x[0])
        top = [gid for (_, gid) in ranked[:k]]
        results = [combined_chunks[i] for i in top if i < len(combined_chunks)]
        return results, embed_tokens

    return [], embed_tokens

# ========================= RAG / GPT RESPONSE =========================
def generate_response(user_input: str, history: List[Dict[str, str]]):
    top_chunks, embedding_tokens = search_chunks(user_input)
    context_text = "\n\n".join([
        f"Source: {c.get('source','')}\nContent: {c.get('text','')}" for c in top_chunks
    ]) if top_chunks else "No context available"

    history_text = "\n".join([
        f"User: {h['user']}\nAssistant: {h['bot']}" for h in history[-3:]
    ]) if history else ""

    prompt = f"""
Role:
You are a helpful, accurate assistant responding ONLY from provided context.

Context:
{context_text}

Recent conversation:
{history_text}

Customer Question: {user_input}

Fallback Instructions:
If answer is NOT found in the context or history, respond EXACTLY:

"I'm sorry, but [specific information] is not available with me. For details, please contact us via https://nivatier.com/contact/"

Do NOT guess or reword the fallback.
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )
        answer = resp['choices'][0]['message']['content']
        usage = resp.get('usage', {})
        return answer, usage.get('prompt_tokens',0), usage.get('completion_tokens',0), usage.get('total_tokens',0), embedding_tokens
    except Exception as e:
        return f"Error: {e}", 0, 0, 0, 0

# ========================= ENDPOINT =========================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Message required")

    response, in_toks, out_toks, total_toks, embed_toks = generate_response(
        user_input, [msg.dict() if hasattr(msg,"dict") else msg for msg in req.history]
    )

    # Cost calculation (example pricing)
    chat_cost = (in_toks / 1000) * 0.01 + (out_toks / 1000) * 0.03
    embed_cost = (embed_toks / 1000) * 0.0001
    total_cost = chat_cost + embed_cost

    log_to_google_sheet([
        datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        req.clientId,
        user_input,
        response,
        in_toks, out_toks, total_toks, embed_toks,
        round(embed_cost,6), round(chat_cost,6), round(total_cost,6)
    ])

    return {"reply": response}

# ========================= STARTUP =========================
@app.on_event("startup")
async def startup_event():
    print("Loading vector databases...")
    if not load_vector_dbs():
        print("❌ Failed to load vector databases. Make sure embed_scraper.py was run.")

# ========================= RUN =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
