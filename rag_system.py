# %%
"""
Simple RAG System with Free LLM
No API keys required - runs completely locally
A good starter template for students to build off of
"""

import os
import torch
from typing import List, Dict
from pathlib import Path

# PDF processing
from pypdf import PdfReader

# Embeddings (free, local)
from sentence_transformers import SentenceTransformer

# Vector store (free, local)
import faiss
import numpy as np

# LLM (free, local)
from transformers import pipeline

# UI
import gradio as gr


class SimpleRAG:
    def __init__(self, pdf_folder: str = "documents"):
        """
        Initialize RAG system
        Args:
            pdf_folder: Folder containing PDF files to process
        """
        print("Initializing RAG System...")
        self.pdf_folder = pdf_folder
        self.chunks = []
        self.chunk_embeddings = None
        self.index = None

        # Initialize models
        print("Loading embedding model (this may take a minute first time)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading language model (this may take a few minutes first time)...")
        # Using smaller model that runs on CPU
        self.llm = pipeline(
            "text-generation",
            model="microsoft/phi-2",  # 2.7B parameter model, runs on most computers
            torch_dtype=torch.float32,
            device="cpu",
            max_new_tokens=200,
        )

        print("RAG System initialized!")

    def load_pdfs(self) -> List[str]:
        """
        Load and extract text from all PDFs in the documents folder

        IMPORTANT: Put your PDF files in the 'documents' folder!
        """
        all_text = []
        pdf_folder = Path(self.pdf_folder)

        # Create folder if it doesn't exist
        pdf_folder.mkdir(exist_ok=True)

        pdf_files = list(pdf_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"\n⚠️  No PDF files found in '{self.pdf_folder}' folder!")
            print(
                f"Please add PDF files to the '{self.pdf_folder}' folder and restart.\n"
            )
            return []

        print(f"\nFound {len(pdf_files)} PDF files:")

        for pdf_path in pdf_files:
            print(f"  - Processing: {pdf_path.name}")
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                all_text.append(text)
                print(f"    ✓ Extracted {len(reader.pages)} pages")
            except Exception as e:
                print(f"    ✗ Error processing {pdf_path.name}: {e}")

        return all_text

    def chunk_text(self, texts: List[str], chunk_size: int = 500) -> List[str]:
        """
        Split text into smaller chunks for processing
        """
        chunks = []
        for text in texts:
            # Simple chunking by character count
            words = text.split()
            current_chunk = []
            current_length = 0

            for word in words:
                current_chunk.append(word)
                current_length += len(word) + 1

                if current_length >= chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

            if current_chunk:
                chunks.append(" ".join(current_chunk))

        print(f"Created {len(chunks)} text chunks")
        return chunks

    def create_embeddings(self, chunks: List[str]):
        """
        Convert text chunks to vector embeddings
        """
        print("Creating embeddings for chunks...")
        embeddings = self.embedder.encode(chunks, show_progress_bar=True)
        return embeddings

    def build_index(self, embeddings):
        """
        Create FAISS index for fast similarity search
        """
        print("Building search index...")
        dimension = embeddings.shape[1]

        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype("float32"))

        print(f"Index built with {index.ntotal} vectors")
        return index

    def setup_knowledge_base(self):
        """
        Main setup function - processes PDFs and creates searchable index
        """
        print("\n" + "=" * 50)
        print("SETTING UP KNOWLEDGE BASE")
        print("=" * 50)

        # Step 1: Load PDFs
        texts = self.load_pdfs()
        if not texts:
            return False

        # Step 2: Chunk texts
        self.chunks = self.chunk_text(texts)

        # Step 3: Create embeddings
        self.chunk_embeddings = self.create_embeddings(self.chunks)

        # Step 4: Build index
        self.index = self.build_index(self.chunk_embeddings)

        print("\n✓ Knowledge base ready!")
        print(f"  - Total chunks: {len(self.chunks)}")
        print(f"  - Embedding dimension: {self.chunk_embeddings.shape[1]}")
        print("=" * 50 + "\n")

        return True

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Find the most relevant chunks for a query
        """
        # Embed the query
        query_embedding = self.embedder.encode([query])

        # Search the index
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)

        # Get the relevant chunks
        relevant_chunks = [self.chunks[i] for i in indices[0]]

        return relevant_chunks

    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generate answer using retrieved context
        """
        # Create prompt with context
        context_text = "\n\n".join(context)

        prompt = f"""Based on the following context, answer the question.
If the answer is not in the context, say "I don't have that information."

Context:
{context_text}

Question: {query}

Answer:"""

        # Generate response
        try:
            response = self.llm(
                prompt, max_new_tokens=200, do_sample=True, temperature=0.7
            )
            return response[0]["generated_text"].split("Answer:")[-1].strip()
        except Exception as e:
            return f"Error generating response: {e}"

    def ask(self, question: str) -> Dict:
        """
        Main RAG pipeline - retrieve and generate
        """
        if not self.index:
            return {"answer": "Please set up the knowledge base first!", "sources": []}

        # Retrieve relevant chunks
        relevant_chunks = self.retrieve(question)

        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)

        return {"answer": answer, "sources": relevant_chunks}


def create_gradio_interface(rag_system):
    """
    Create web interface for the RAG system
    """

    def chatbot_response(message, history):
        result = rag_system.ask(message)

        # Format response with sources
        response = f"**Answer:** {result['answer']}\n\n"
        response += "**Sources:**\n"
        for i, source in enumerate(result["sources"], 1):
            response += f"\n{i}. {source[:200]}...\n"

        return response

    # Create Gradio interface
    demo = gr.ChatInterface(
        chatbot_response,
        title="📚 PDF RAG Chatbot",
        description="Ask questions about your PDF documents!",
        examples=[
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key findings?",
        ],
        theme="soft",
    )

    return demo


def main():
    """
    Main function to run the RAG system
    """
    print("\n" + "🚀 SIMPLE RAG SYSTEM WITH FREE LLM 🚀".center(50))
    print("=" * 50)

    # Initialize RAG system
    rag = SimpleRAG(pdf_folder="documents")

    # Setup knowledge base from PDFs
    if not rag.setup_knowledge_base():
        print("\n❌ No PDFs found. Please add PDF files to the 'documents' folder.")
        print("\nFolder structure should be:")
        print("  simple-rag-project/")
        print("    ├── documents/         <-- PUT YOUR PDFs HERE")
        print("    │   └── your_file.pdf")
        print("    └── rag_system.py")
        return

    # Create and launch web interface
    print("\n🌐 Launching web interface...")
    demo = create_gradio_interface(rag)
    demo.launch(share=False)  # share=False for local only


if __name__ == "__main__":
    main()

# %%
