# app.py
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

# ========== CONFIGURATION ==========
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

VECTOR_DB_DIR = "vector_db"
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
SHEET_NAME = "Sheet1"
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")

if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDS_JSON environment variable")

try:
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    sheets_service = build("sheets", "v4", credentials=creds)
    sheet = sheets_service.spreadsheets()
except Exception as e:
    raise RuntimeError(f"Failed to load Google credentials: {str(e)}")

# ========== FastAPI App Initialization ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Global State ==========
faiss_index = None
combined_chunks: List[Dict] = []
index_dimension = None

# ========== Pydantic Models ==========
class ChatMessage(BaseModel):
    user: str
    bot: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    clientId: str

class ChatResponse(BaseModel):
    reply: str

# ========== Utilities ==========
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


def count_tokens(text: str, model: str = "gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_openai_embedding(text: str) -> Tuple[List[float], int]:
    """
    Uses OpenAI embeddings API (text-embedding-ada-002) and returns (embedding, tokens_used_for_embedding)
    """
    resp = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return resp['data'][0]['embedding'], resp['usage']['total_tokens']

# ========== FAISS Loading & Search ==========
def load_all_vector_dbs() -> bool:
    """
    Load all .index files from VECTOR_DB_DIR and their metadata.
    Merges them into a single FAISS index if multiple are found.
    """
    global faiss_index, combined_chunks, index_dimension

    if not os.path.exists(VECTOR_DB_DIR):
        print(f"Vector DB directory '{VECTOR_DB_DIR}' not found")
        return False

    db_files = [f for f in os.listdir(VECTOR_DB_DIR) if f.endswith('.index')]
    if not db_files:
        print("No FAISS index files found in vector_db/")
        return False

    combined_chunks = []
    indexes = []
    dims = set()

    for db_file in sorted(db_files):
        base = db_file[:-len(".index")]
        index_path = os.path.join(VECTOR_DB_DIR, db_file)
        meta_path = os.path.join(VECTOR_DB_DIR, f"{base}_metadata.json")

        print(f"Loading FAISS index: {index_path}")
        try:
            index = faiss.read_index(index_path)
        except Exception as e:
            print(f"Failed to read index {index_path}: {e}")
            continue

        # Determine dimension from index (works for flat indexes)
        try:
            # index.d is available for some index types
            dim = faiss.vector_to_array(index.reconstruct(0)).shape[0] if index.ntotal > 0 else None
        except Exception:
            dim = None

        # fallback: try to get d from index description if possible
        try:
            ddesc = str(index)
            # This is heuristic and may not always work. If not available, dimension will be set below.
            if "d=" in ddesc:
                # parse e.g. 'IndexFlatL2(d=1536)'
                import re
                m = re.search(r"d\s*=\s*(\d+)", ddesc)
                if m:
                    dim = int(m.group(1))
        except Exception:
            pass

        if dim is not None:
            dims.add(dim)

        indexes.append(index)

        # load metadata
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if "chunks" in metadata:
                    combined_chunks.extend(metadata["chunks"])
        else:
            print(f"Warning: metadata {meta_path} not found for index {index_path}")

    # Validate dimensions
    if len(dims) == 0:
        # try to infer from first chunk embedding in metadata
        found_dim = None
        for ch in combined_chunks:
            if "embedding" in ch and isinstance(ch["embedding"], list):
                found_dim = len(ch["embedding"])
                break
        if found_dim:
            index_dimension = found_dim
        else:
            print("Could not determine embedding dimension. Aborting load.")
            return False
    elif len(dims) == 1:
        index_dimension = dims.pop()
    else:
        # multiple dimensions found across indexes -> incompatible
        print("Incompatible index dimensions found across FAISS indexes:", dims)
        return False

    # Merge indexes (if multiple)
    if len(indexes) == 1:
        faiss_index = indexes[0]
    else:
        try:
            # concat_indexes requires all indexes to be the same type/dimension
            faiss_index = faiss.merge_indexes(indexes) if hasattr(faiss, "merge_indexes") else faiss.concat_indexes(indexes)
        except Exception as e:
            print("Failed to merge FAISS indexes. Attempting sequential search fallback.", e)
            # fallback: keep list of indexes and do sequential search (we'll handle this below)
            faiss_index = None
            app.state._faiss_indexes = indexes  # attach to app state for fallback
            app.state._use_sequential_search = True

    print(f"Loaded {len(combined_chunks)} chunks from {len(db_files)} indexes. Dimension: {index_dimension}")
    return True


def search_similar_chunks(query: str, k: int = 5) -> Tuple[List[Dict], int]:
    """
    Returns top-k chunk dicts and embedding token usage.
    If merged faiss_index exists, uses single search; otherwise uses sequential search across indexes.
    """
    global faiss_index, combined_chunks, index_dimension

    # create query embedding
    q_emb, embed_tokens = get_openai_embedding(query)
    q_vec = np.array(q_emb).astype('float32').reshape(1, -1)

    if faiss_index is not None:
        # Search merged index
        D, I = faiss_index.search(q_vec, k)
        indices = I[0].tolist()
        results = []
        for idx in indices:
            if idx < len(combined_chunks):
                results.append(combined_chunks[idx])
        return results, embed_tokens

    # fallback: sequential search across indexes stored in app.state._faiss_indexes
    if getattr(app.state, "_use_sequential_search", False):
        all_scores = []
        all_idxs = []
        base_offset = 0
        chunks_accum = []
        # gather chunks for each index from metadata order: we already combined chunks in load_all_vector_dbs()
        # We'll compute nearest neighbors by searching each index and collecting (distance, global_index)
        for index in getattr(app.state, "_faiss_indexes", []):
            try:
                D, I = index.search(q_vec, k)
                for dist, local_idx in zip(D[0], I[0]):
                    if local_idx == -1:
                        continue
                    global_idx = base_offset + int(local_idx)
                    all_scores.append(float(dist))
                    all_idxs.append(global_idx)
            except Exception:
                pass
            # increment base_offset by the number of vectors in this index
            base_offset += index.ntotal

        # sort best k by distance (L2, lower is better)
        ranked = sorted(zip(all_scores, all_idxs), key=lambda x: x[0])
        top = [gid for (_, gid) in ranked[:k]]
        results = [combined_chunks[i] for i in top if i < len(combined_chunks)]
        return results, embed_tokens

    # If nothing available
    return [], embed_tokens

# ========== RAG / Chat Generation ==========
def generate_response(user_input: str, history: List[Dict[str, str]]):
    relevant_chunks, embedding_tokens = search_similar_chunks(user_input)
    if not relevant_chunks:
        context = None
    else:
        context_parts = [
            f"Source: {chunk.get('source','')}\nContent: {chunk.get('text','')}"
            for chunk in relevant_chunks
        ]
        context = "\n\n".join(context_parts)

    history_parts = [
        f"User: {msg['user']}\nAssistant: {msg['bot']}"
        for msg in history[-3:]
    ]
    history_text = "\n".join(history_parts)

    prompt = f"""
    Role:
    You are a helpful, accurate assistant designed to respond using only the information provided.

    Context:
    {context}

    Recent conversation:
    {history_text}

    Customer Question: {user_input}

    Fallback Instructions (MANDATORY):
    If the answer to a question is NOT found in the {context} or in the {history_text}, respond with this exact fallback format:

    "I'm sorry, but [specific information] is not available with me. For detailed information, please contact us directly via the contact page.
    Source: https://nivatier.com/contact/"

    Do NOT reword the fallback. Do NOT guess.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )
        choice = response['choices'][0]['message']['content']
        usage = response.get('usage', {})
        return choice, usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), usage.get('total_tokens', 0), embedding_tokens
    except Exception as e:
        return f"Error: {str(e)}", 0, 0, 0, 0

# ========== FastAPI Endpoints ==========
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_input = request.message.strip()
        client_id = request.clientId
        if not user_input:
            raise HTTPException(status_code=400, detail="Message is required")

        response, input_toks, output_toks, total_toks, embed_toks = generate_response(
            user_input, [msg.dict() if hasattr(msg, "dict") else msg for msg in request.history]
        )

        chat_cost = (input_toks / 1000) * 0.01 + (output_toks / 1000) * 0.03
        embed_cost = (embed_toks / 1000) * 0.0001
        total_cost = chat_cost + embed_cost

        log_to_google_sheet([
            datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            client_id,
            user_input,
            response,
            input_toks,
            output_toks,
            total_toks,
            embed_toks,
            round(embed_cost, 6),
            round(chat_cost, 6),
            round(total_cost, 6),
        ])

        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Startup ==========
@app.on_event("startup")
async def on_startup():
    print("Loading vector databases...")
    ok = load_all_vector_dbs()
    if not ok:
        print("Failed to load vector databases. Make sure you ran embed_scraper.py to generate .index files.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
