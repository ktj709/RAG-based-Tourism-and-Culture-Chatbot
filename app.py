from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embed_store import EmbedStore
from retrieve import retrieve_top_chunks
from generate import generate_answer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize FastAPI app
app = FastAPI(title="Tourism Chatbot API", version="1.0")

# Load or create FAISS index
store = EmbedStore()

# Try loading pre-built FAISS index and metadata
try:
    store.load()
    print("✅ Loaded existing FAISS index.")
except FileNotFoundError:
    print("⚠️ No existing FAISS index found. Please build one using main.py first.")


# Define request schema
class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "Welcome to the Tourism Chatbot API 🚀"}


@app.post("/query")
def query_tourism_bot(request: QueryRequest):
    """
    Handles user queries using RAG + Gemini 2.0 fallback.
    """
    query = request.query

    try:
        # Retrieve top relevant chunks
        top_chunks = retrieve_top_chunks(store, query)

        if not top_chunks:
            raise ValueError("No relevant documents found")

        # Generate answer using contextual retrieval
        answer = generate_answer(query, top_chunks)
        return {"response": answer}

    except Exception as e:
        # Fallback: Use Gemini directly for general queries
        print(f"⚠️ Fallback triggered: {e}")
        fallback_prompt = f"You are a helpful travel assistant. Answer the following query:\n\n{query}"
        fallback_model = genai.GenerativeModel("gemini-2.0-flash")
        response = fallback_model.generate_content(fallback_prompt)
        return {"response": response.text or "Sorry, I couldn't find any relevant information."}


@app.get("/health")
def health_check():
    return {"status": "ok"}
