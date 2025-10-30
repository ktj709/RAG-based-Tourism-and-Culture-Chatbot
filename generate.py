import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

def generate_answer(query, top_chunks, max_output_tokens=500):
    """
    Generates an answer using retrieved context if relevant.
    If the context seems unrelated to the query (e.g., about another city),
    automatically falls back to Gemini’s general knowledge.
    """

    # 🧩 1️⃣ Combine retrieved content
    context = "\n".join([chunk.page_content for chunk in top_chunks]) if top_chunks else ""

    # 🧠 2️⃣ If no context at all — direct fallback
    if not context.strip():
        print("⚠️ No FAISS context found. Using Gemini fallback...")
        return _fallback_to_gemini(query, max_output_tokens)

    # 🧠 3️⃣ Ask Gemini to judge if the context is relevant
    judge_prompt = (
        f"You are an AI assistant that checks if context is relevant to a question.\n"
        f"Question: {query}\n"
        f"Context: {context[:1500]}\n\n"  # limit to avoid huge inputs
        f"Answer only 'yes' if the context directly relates to the question, else 'no'."
    )

    judge_resp = model.generate_content(judge_prompt)
    judge_text = judge_resp.text.lower().strip()

    # 🧠 4️⃣ If Gemini thinks the context is irrelevant, fallback
    if "no" in judge_text:
        print("⚠️ Irrelevant context detected. Using Gemini fallback...")
        return _fallback_to_gemini(query, max_output_tokens)

    # 🧠 5️⃣ Otherwise, use the retrieved chunks
    prompt = (
        "You are a helpful tourism and culture guide.\n"
        "Use the following context to answer the question.\n"
        "If it helps, you may add your general knowledge to make the answer more complete.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    response = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_output_tokens,
            "temperature": 0.3
        }
    )
    return response.text.strip()


def _fallback_to_gemini(query, max_output_tokens):
    """
    Handles general fallback using Gemini Flash.
    """
    fallback_prompt = (
        "You are a friendly tourism and culture guide. "
        "Use your knowledge to answer this question in detail:\n\n"
        f"{query}"
    )
    response = model.generate_content(
        fallback_prompt,
        generation_config={
            "max_output_tokens": max_output_tokens,
            "temperature": 0.7
        }
    )
    return response.text.strip()
