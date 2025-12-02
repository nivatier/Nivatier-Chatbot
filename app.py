
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict
import numpy as np
import os
import json
import openai
import tiktoken
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
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

# GOOGLE_CREDS_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

# Google Sheets API Setup
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
# Load Google service account credentials from environment variable
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")

if not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing GOOGLE_CREDS_JSON environment variable")

try:
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    sheets_service = build("sheets", "v4", credentials=creds)
except Exception as e:
    raise RuntimeError(f"Failed to load Google credentials: {str(e)}")

sheet = sheets_service.spreadsheets()

# ========== FastAPI App Initialization ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Global State ==========
combined_embeddings = None
all_chunks = []

# ========== Pydantic Models ==========
class ChatMessage(BaseModel):
    user: str
    bot: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]  # Frontend sends chat history
    clientId: str  # Add this line

class ChatResponse(BaseModel):
    reply: str

# ========== Utility Functions ==========
def log_to_google_sheet(row_data: List):
    sheet.values().append(
        spreadsheetId=SHEET_ID,
        range=f"{SHEET_NAME}!A1",
        valueInputOption="USER_ENTERED",
        body={"values": [row_data]}
    ).execute()

def count_tokens(text: str, model: str = "gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding'], response['usage']['total_tokens']

# ========== Core Functions ==========
def load_all_vector_dbs():
    global combined_embeddings, all_chunks

    if not os.path.exists(VECTOR_DB_DIR):
        print(f"Vector DB directory '{VECTOR_DB_DIR}' not found")
        return False

    db_files = [f for f in os.listdir(VECTOR_DB_DIR) if f.endswith('.npy')]
    if not db_files:
        print("No vector databases found")
        return False

    all_chunks.clear()
    embeddings_list = []

    for db_file in db_files:
        db_name = db_file.replace('.npy', '')
        db_path = os.path.join(VECTOR_DB_DIR, db_name)

        embeddings = np.load(f"{db_path}.npy")
        embeddings_list.append(embeddings)

        with open(f"{db_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
            all_chunks.extend(metadata['chunks'])

    combined_embeddings = np.vstack(embeddings_list)
    return True

def search_similar_chunks(query, k=5):
    global combined_embeddings, all_chunks

    if combined_embeddings is None:
        print("Knowledge base not initialized")
        return []

    query_embedding, embedding_tokens = get_openai_embedding(query)
    similarities = cosine_similarity([query_embedding], combined_embeddings)[0]
    top_k_indices = similarities.argsort()[::-1][:k]
    return [all_chunks[i] for i in top_k_indices], embedding_tokens

def generate_response(user_input, history: List[Dict[str, str]]):
    relevant_chunks, embedding_tokens = search_similar_chunks(user_input)
    if not relevant_chunks:
        context = None
        # return "I couldn't find relevant information to answer your question.", 0, 0, 0, embedding_tokens
    else:
        context_parts = [
            f"Source: {chunk['source']}\nContent: {chunk['text']}"
            for chunk in relevant_chunks
        ]
        context = "\n\n".join(context_parts)

    history_parts = [
        f"User: {msg['user']}\nAssistant: {msg['bot']}"
        for msg in history[-3:]
    ]
    history_text = "\n".join(history_parts)
    #print(history_text)

    prompt = f"""
    Role:
    You are a helpful, accurate assistant for DukeToms, designed to respond using only the information provided.

    Context:
    {context}

    Recent conversation:
    {history_text}

    Customer Question: {user_input}

    Fallback Instructions (Mandatory):
    If the answer to a question is NOT found in the {context} or in the {history_text}, respond with this **exact fallback format**:

    "I'm sorry, but [specific information] is not available with me. For detailed information, please contact us directly via the contact page.  
    Source: https://duketoms.com/contact/"

    Just replace [specific information] with the topic (e.g., “pricing”, “return policy”, “delivery info”).

    ⚠️ Do NOT reword or rephrase this fallback. Do NOT guess. Always follow this format word-for-word when the context is missing.  
    ⚠️ Use line breaks for: lists, category names, or multiple links.

    Tone: 80% Professional Helper, 20% Playful Buddy (warm tone for informal chats or follow-ups).

    Guidelines:
    1. ONLY use the provided context – no assumptions or outside info.
    2. Ask for the user's name in the first interaction to personalize the chat.
    3. Use their name in future messages to maintain a friendly tone.
    4. If answer is missing, ALWAYS use the exact fallback above.
    5. BE BRIEF.
       – 1–2 lines only.
       – Don’t list features unless asked.
       – End with: “Let me know if you'd like more info.”
    6. ALWAYS include a **Source URL** for every answer, even if it’s generic:  
       Source: https://duketoms.com/page-name  
       (If page not available, use: https://duketoms.com/contact/)
    7. Use line breaks for category lists, like:  
       a. Hygiene Solutions  
       b. Microbial and Safety Testing  
       c. Specialized Industry Solutions
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
        usage = response['usage']
        return choice, usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens'], embedding_tokens
    except Exception as e:
        return f"Error: {str(e)}", 0, 0, 0, 0


# ========== FastAPI Endpoints ==========
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_input = request.message.strip()
        client_id = request.clientId  # Access client ID here
        print(request)
        #print(request.history)
        if not user_input:
            raise HTTPException(status_code=400, detail="Message is required")

        response, input_toks, output_toks, total_toks, embed_toks = generate_response(
            user_input, [msg.dict() for msg in request.history]
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
        return f"Error: {str(e)}"

# ========== Startup Events ==========
@app.on_event("startup")
async def on_startup():
    print("Loading vector databases...")
    if not load_all_vector_dbs():
        print("Failed to load vector databases")
