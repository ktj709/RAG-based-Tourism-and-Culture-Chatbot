import streamlit as st
from embed_store import EmbedStore
from retrieve import retrieve_top_chunks
from generate import generate_answer
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Page configuration
st.set_page_config(
    page_title="Tourism & Culture Guide Chatbot",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #616161;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
    }
    .bot-message {
        background-color: #F5F5F5;
        border-left: 5px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌍 Tourism & Culture Guide Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask me anything about tourism destinations and cultural information!</p>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.store_loaded = False

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This chatbot uses RAG (Retrieval Augmented Generation) to answer your questions about:
    - 🏛️ Tourist destinations
    - 🎨 Cultural information
    - 🗺️ Travel recommendations
    - 🍽️ Local cuisine and attractions
    """)
    
    st.header("⚙️ Settings")
    
    # Load/Build Index
    if st.button("🔄 Load Knowledge Base"):
        with st.spinner("Loading knowledge base..."):
            try:
                store = EmbedStore()
                store.load()
                st.session_state.store = store
                st.session_state.store_loaded = True
                st.success(f"✅ Loaded {len(store.chunks)} chunks successfully!")
            except FileNotFoundError:
                st.error("⚠️ No knowledge base found. Please run `main.py` first to build the index.")
            except Exception as e:
                st.error(f"❌ Error loading knowledge base: {str(e)}")
    
    if st.session_state.store_loaded:
        st.success("✅ Knowledge base is loaded")
        st.metric("Total Chunks", len(st.session_state.store.chunks))
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Powered by Google Gemini 2.0 Flash")

# Main chat interface
st.header("💬 Chat")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message"><strong>Bot:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me about tourism and culture..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{prompt}</div>', unsafe_allow_html=True)
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            if st.session_state.store_loaded and st.session_state.store:
                # Use RAG with loaded knowledge base
                top_chunks = retrieve_top_chunks(st.session_state.store, prompt)
                answer = generate_answer(prompt, top_chunks)
            else:
                # Fallback to direct Gemini query
                model = genai.GenerativeModel("gemini-2.0-flash")
                fallback_prompt = f"You are a helpful tourism and culture guide. Answer the following query:\n\n{prompt}"
                response = model.generate_content(fallback_prompt)
                answer = response.text or "Sorry, I couldn't find any relevant information."
                st.info("ℹ️ Using general knowledge (knowledge base not loaded)")
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display bot response
            st.markdown(f'<div class="chat-message bot-message"><strong>Bot:</strong><br>{answer}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔒 Secure & Private")
with col2:
    st.caption("⚡ Fast Responses")
with col3:
    st.caption("🌐 Multi-language Support")
