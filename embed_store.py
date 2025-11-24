import os
import pickle
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

class Document:
    """Simple document class to replace langchain Document"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class EmbedStore:
    """Lightweight embedding store using Google's Embedding API instead of local models"""
    
    def __init__(self, meta_path='chunks_meta.pkl'):
        self.chunks: List[Document] = []
        self.meta_path = meta_path
        self.embeddings_cache = {}  # Cache embeddings to reduce API calls

    def build_index(self, chunks: List[Document], rebuild: bool = True):
        """Store chunks without building heavy FAISS index"""
        self.chunks = chunks
        print(f"✅ Stored {len(chunks)} chunks (using Google Embeddings API)")

    def save(self, meta_path: str = None):
        meta_path = meta_path or self.meta_path
        with open(meta_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Saved chunks metadata to {meta_path}")

    def load(self, meta_path: str = None):
        meta_path = meta_path or self.meta_path
        if not os.path.exists(meta_path):
            raise FileNotFoundError('Metadata file not found')
        with open(meta_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"✅ Loaded {len(self.chunks)} chunks")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using Google's API with caching"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embedding = result['embedding']
        self.embeddings_cache[text] = embedding
        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def search(self, query: str, k: int = 5):
        """Search using Google embeddings and cosine similarity"""
        if not self.chunks:
            raise RuntimeError('No chunks loaded')
        
        # Get query embedding
        query_result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_result['embedding']
        
        # Calculate similarities
        similarities = []
        for i, chunk in enumerate(self.chunks):
            chunk_embedding = self._get_embedding(chunk.page_content)
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((similarity, i, chunk))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, _, chunk in similarities[:k]]
