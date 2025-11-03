#%%
"""
RAG Chatbot with HuggingFace API
A simple, student-friendly implementation using HuggingFace's InferenceClient
"""

import os
from typing import List, Dict
from pathlib import Path

# PDF processing
from pypdf import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector store
import faiss
import numpy as np

# API requests
import requests

# UI
import gradio as gr


class RAGChatbot:
    """
    A complete RAG system using HuggingFace's free Inference API
    """

    def __init__(self, hf_token: str, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize the RAG chatbot

        Args:
            hf_token: HuggingFace API token
            model_name: Model to use for generation
        """
        print("=" * 60)
        print("🚀 INITIALIZING RAG CHATBOT")
        print("=" * 60)

        self.hf_token = hf_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"

        # Setup API headers
        print(f"\n✓ Setting up HuggingFace Inference API")
        print(f"  Model: {model_name}")
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        print("  ✓ API configured")

        # Initialize embedding model (runs locally)
        print("\n✓ Loading embedding model (local)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("  ✓ Embedding model loaded")

        # Initialize storage
        self.chunks = []
        self.chunk_embeddings = None
        self.index = None

        print("\n" + "=" * 60)
        print("✅ RAG CHATBOT READY")
        print("=" * 60 + "\n")

    def load_pdfs(self, pdf_folder: str = "documents") -> List[str]:
        """
        Load and extract text from all PDFs in folder

        Args:
            pdf_folder: Path to folder containing PDFs

        Returns:
            List of text strings, one per PDF
        """
        print(f"\n📄 Loading PDFs from '{pdf_folder}/'...")

        all_text = []
        pdf_folder = Path(pdf_folder)

        # Create folder if it doesn't exist
        pdf_folder.mkdir(exist_ok=True)

        pdf_files = list(pdf_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"  ⚠️  No PDF files found in '{pdf_folder}/'")
            return []

        print(f"  Found {len(pdf_files)} PDF file(s)")

        for pdf_path in pdf_files:
            print(f"  Processing: {pdf_path.name}")
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                all_text.append(text)
                print(f"    ✓ Extracted {len(reader.pages)} pages")
            except Exception as e:
                print(f"    ✗ Error: {e}")

        return all_text

    def chunk_text(self, texts: List[str], chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split texts into smaller chunks with overlap

        Args:
            texts: List of text strings
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        print(f"\n✂️  Chunking text (size={chunk_size}, overlap={overlap})...")

        chunks = []

        for text in texts:
            # Split into sentences
            sentences = text.replace('\n', ' ').split('. ')

            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_length = len(sentence) + 2

                if current_length + sentence_length > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append(chunk_text)

                    # Start new chunk with overlap
                    if overlap > 0:
                        overlap_text = chunk_text[-overlap:]
                        current_chunk = [overlap_text + sentence]
                        current_length = len('. '.join(current_chunk))
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            # Add remaining chunk
            if current_chunk:
                chunk_text = '. '.join(current_chunk)
                if not chunk_text.endswith('.'):
                    chunk_text += '.'
                chunks.append(chunk_text)

        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"  ✓ Avg length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

        return chunks

    def create_vector_store(self, chunks: List[str]):
        """
        Create embeddings and FAISS index

        Args:
            chunks: List of text chunks
        """
        print(f"\n🔢 Creating vector store...")

        # Create embeddings
        print(f"  Encoding {len(chunks)} chunks...")
        self.chunk_embeddings = self.embedder.encode(
            chunks,
            show_progress_bar=True,
            batch_size=32
        )

        # Build FAISS index
        print(f"  Building FAISS index...")
        dimension = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.chunk_embeddings.astype('float32'))

        print(f"  ✓ Vector store ready ({self.index.ntotal} vectors)")

    def setup_knowledge_base(self, pdf_folder: str = "documents"):
        """
        Main setup: load PDFs, chunk, and create vector store

        Args:
            pdf_folder: Folder containing PDF files
        """
        # Load PDFs
        texts = self.load_pdfs(pdf_folder)
        if not texts:
            print("\n❌ No PDFs loaded. Please add PDFs to the 'documents/' folder.")
            return False

        # Chunk texts
        self.chunks = self.chunk_text(texts)

        # Create vector store
        self.create_vector_store(self.chunks)

        print("\n" + "=" * 60)
        print("✅ KNOWLEDGE BASE READY")
        print(f"   Total chunks: {len(self.chunks)}")
        print(f"   Embedding dim: {self.chunk_embeddings.shape[1]}")
        print("=" * 60 + "\n")

        return True

    def retrieve(self, query: str, top_k: int = 5) -> tuple:
        """
        Retrieve most relevant chunks for a query

        Args:
            query: Search query
            top_k: Number of chunks to retrieve

        Returns:
            Tuple of (chunks, similarities)
        """
        # Encode query
        query_embedding = self.embedder.encode([query])

        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, len(self.chunks))
        )

        # Convert distances to similarities
        similarities = 1 / (1 + distances[0])

        # Get chunks
        relevant_chunks = [self.chunks[i] for i in indices[0]]

        return relevant_chunks, similarities

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using HuggingFace QA API

        Args:
            query: User's question
            context_chunks: Relevant context chunks

        Returns:
            Generated answer
        """
        # Combine context
        context_text = " ".join(context_chunks)

        # Truncate if too long (QA models have limits)
        max_context_length = 4000
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."

        try:
            # Call HuggingFace QA API
            payload = {
                "inputs": {
                    "question": query,
                    "context": context_text
                }
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                # QA model returns: {'score': 0.xx, 'answer': 'text', 'start': x, 'end': y}
                if isinstance(result, dict) and 'answer' in result:
                    answer = result['answer'].strip()
                    confidence = result.get('score', 0)

                    # If confidence is too low, say so
                    if confidence < 0.1:
                        return f"{answer} (Low confidence - answer might not be accurate)"

                    return answer
                else:
                    return f"Unexpected response format: {str(result)[:200]}"

            elif response.status_code == 503:
                return "⏳ Model is loading... Please wait 30-60 seconds and try again."
            elif response.status_code == 404:
                return f"❌ Model '{self.model_name}' not found."
            elif response.status_code == 401:
                return "❌ Invalid HuggingFace token. Please check your token."
            else:
                return f"API Error ({response.status_code}): {response.text[:200]}"

        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Try again."
        except Exception as e:
            return f"Error: {str(e)}"

    def ask(self, question: str, top_k: int = 5) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with answer, sources, and similarities
        """
        if not self.index:
            return {
                "answer": "❌ Knowledge base not set up. Run setup_knowledge_base() first.",
                "sources": [],
                "similarities": []
            }

        # Retrieve relevant chunks
        relevant_chunks, similarities = self.retrieve(question, top_k)

        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)

        return {
            "answer": answer,
            "sources": relevant_chunks,
            "similarities": similarities
        }


def create_gradio_interface(rag: RAGChatbot):
    """
    Create Gradio web interface

    Args:
        rag: RAGChatbot instance

    Returns:
        Gradio interface
    """
    def chatbot_response(message, history):
        result = rag.ask(message, top_k=5)

        # Format response
        response = f"{result['answer']}\n\n"

        if result['sources']:
            response += "---\n**📚 Sources:**\n"
            for i, (source, sim) in enumerate(
                zip(result['sources'][:3], result['similarities'][:3]),
                1
            ):
                source_preview = source[:150].replace('\n', ' ')
                response += f"\n{i}. (Similarity: {sim:.2f}) {source_preview}...\n"

        return response

    interface = gr.ChatInterface(
        chatbot_response,
        title="📚 RAG Chatbot with HuggingFace",
        description="Ask questions about your PDF documents!",
        examples=[
            "What is this document about?",
            "Summarize the main points",
            "What are the key concepts?",
        ],
        theme="soft",
    )

    return interface


def main():
    """
    Main function - run the RAG chatbot
    """
    print("\n" + "=" * 60)
    print("RAG CHATBOT - HuggingFace Edition")
    print("=" * 60 + "\n")

    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("⚠️  HF_TOKEN environment variable not set!")
        print("\nPlease set your token:")
        print("  export HF_TOKEN='your_token_here'")
        print("\nOr get your token from: https://huggingface.co/settings/tokens")
        hf_token = input("\nEnter your HuggingFace token: ").strip()

    if not hf_token or hf_token == "your_token_here":
        print("\n❌ No valid token provided. Exiting.")
        return

    # Use the QA model (works on free tier!)
    model_name = "deepset/roberta-base-squad2"
    print(f"\n✓ Using model: {model_name}")
    print("  This model is free and designed for question-answering!")

    # Initialize RAG chatbot
    try:
        rag = RAGChatbot(hf_token=hf_token, model_name=model_name)
    except Exception as e:
        print(f"\n❌ Failed to initialize chatbot: {e}")
        return

    # Setup knowledge base
    if not rag.setup_knowledge_base("documents"):
        return

    # Test question
    print("\n🧪 Testing with a sample question...")
    test_result = rag.ask("What is this document about?")
    print(f"\n📝 Answer: {test_result['answer'][:200]}...")

    # Launch Gradio interface
    print("\n🌐 Launching web interface...")
    print("   Press Ctrl+C to stop")

    interface = create_gradio_interface(rag)
    interface.launch(share=False)


if __name__ == "__main__":
    main()

# %%
