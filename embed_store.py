from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from typing import List
from langchain_core.documents import Document

class EmbedStore:
    def __init__(self, model_name='all-mpnet-base-v2', index_path='faiss_index.bin', meta_path='chunks_meta.pkl'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[Document] = []
        self.index_path = index_path
        self.meta_path = meta_path

    def build_index(self, chunks: List[Document], rebuild: bool = True):
        texts = [c.page_content for c in chunks]
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype='float32')
        d = embeddings.shape[1]
        if rebuild or self.index is None:
            self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        self.chunks = chunks

    def save(self, index_path: str = None, meta_path: str = None):
        index_path = index_path or self.index_path
        meta_path = meta_path or self.meta_path
        if self.index is None:
            raise RuntimeError('No index to save')
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.chunks, f)

    def load(self, index_path: str = None, meta_path: str = None):
        index_path = index_path or self.index_path
        meta_path = meta_path or self.meta_path
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError('Index or metadata file not found')
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.chunks = pickle.load(f)

    def search(self, query: str, k: int = 5):
        q_emb = self.model.encode([query], convert_to_tensor=False)
        q_np = np.array(q_emb, dtype='float32')
        if self.index is None:
            raise RuntimeError('Index not initialized')
        D, I = self.index.search(q_np, k)
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[int(idx)])
        return results
