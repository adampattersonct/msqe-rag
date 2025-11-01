# RAG System Template - Learning Project

*A starter template for building a Retrieval-Augmented Generation (RAG) system using free, local models.*

## Overview

This project provides a foundational RAG system that runs completely locally without requiring API keys. It's designed as a learning template for students to understand RAG architecture and experiment with enhancements.

**Course:** ECON 5502 — Fall 2025

## Current Implementation

The `rag_system.py` script includes:

- **PDF Processing**: Extract text from local PDF documents using PyPDF
- **Embeddings**: Generate vector embeddings with SentenceTransformer (all-MiniLM-L6-v2)
- **Vector Storage**: Fast similarity search using FAISS
- **Language Model**: Text generation with Microsoft Phi-2 (2.7B parameters, CPU-optimized)
- **Web Interface**: Interactive chat UI built with Gradio

All components run locally on your machine - no API keys or cloud services required!

## Setup Instructions after creating virtual environment

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Your Documents:**
   - Create a `documents/` folder in the project directory
   - Place your PDF files in the `documents/` folder

3. **Run the System:**
   ```bash
   python rag_system.py
   ```

4. **Use the Interface:**
   - The Gradio interface will launch in your browser
   - Ask questions about your PDF documents
   - View retrieved sources for each answer

## Project Structure

```
msqe-rag/
├── documents/           # Put your PDF files here
│   └── your_file.pdf
├── rag_system.py        # Main RAG implementation
├── requirements.txt     # Project python dependencies
└── README.md
```

## Potential Enhancements

This template is intentionally kept simple to facilitate learning. Here are some potential features you could add:

### 1. Advanced Language Models
- **Hugging Face API Integration**: Replace the local Phi-2 model with more powerful models via Hugging Face's Inference API
  - Access cutting-edge models like Llama 3, Mistral, or GPT variants
  - Benefit: Better answer quality and reasoning capabilities
  - Trade-off: Requires API key and internet connection

### 2. Web Scraping for PDFs
- **Automated Document Collection**: Scrape PDFs directly from websites
  - Implement web scraping with libraries like BeautifulSoup, Scrapy, or requests
  - Automatically download and process PDFs from government sites, research repositories, etc.
  - Benefit: Scalable data collection without manual downloads
  - Use case: Continuously update knowledge base with new publications

### 3. Vector Database Integration (Scaling Solution)
- **Replace FAISS with Cloud Vector Databases**: An important exercise in building scalable RAG systems
  - **Current limitation**: FAISS stores vectors in memory, data lost when program ends
  - **Pinecone**: Managed vector database service
    - Persistent storage with automatic backups
    - Handles billions of vectors with low latency
    - Built-in metadata filtering and hybrid search
    - Free tier available for learning
  - **Alternatives**: Weaviate, ChromaDB, Qdrant, Milvus
  - **Benefits of migration**:
    - Data persistence across sessions
    - Horizontal scaling for production workloads
    - No need to rebuild index on restart
    - Better performance with large document collections
  - **Learning outcomes**: Understand production-ready vector search architecture

### 4. RAG Evaluation & Comparison Interface
- **Side-by-Side Chatbot Comparison**: Build a dual-interface to evaluate RAG effectiveness
  - **Left panel**: RAG-enabled chatbot (retrieval + generation)
  - **Right panel**: Foundation model only (no retrieval context)
  - **Purpose**: Directly compare how RAG improves responses
  - **Implementation approach**:
    - Modify Gradio interface to show two chat windows simultaneously
    - Send same query to both systems
    - Display responses side-by-side for comparison
  - **What to evaluate**:
    - Factual accuracy (does RAG provide correct info from documents?)
    - Hallucination reduction (does foundation model make up facts?)
    - Answer relevance and specificity
    - Citation and source grounding
  - **Learning outcomes**:
    - Understand the value proposition of RAG systems
    - Learn evaluation methodologies for LLM applications
    - Develop critical thinking about model outputs
  - **Extension ideas**:
    - Add evaluation metrics (BLEU, ROUGE, or custom rubrics)
    - Include human feedback buttons (thumbs up/down)
    - Log comparison results for analysis

### 5. Additional Scalability Improvements
- **Chunking Strategies**: Implement smarter text splitting
  - Semantic chunking based on paragraphs/sections
  - Overlapping chunks to preserve context
- **Batch Processing**: Handle large document collections more efficiently
- **Caching**: Add response caching for frequently asked questions

### 6. Other Ideas
- Add support for multiple document formats (Word, HTML, plain text)
- Implement citation tracking to show exact page numbers
- Add query history and session persistence
- Create evaluation metrics for answer quality
- Build a REST API for programmatic access

## Learning Goals

1. Understand the RAG pipeline: Retrieval → Augmentation → Generation
2. Experiment with different models and parameters
3. Learn about vector embeddings and similarity search
4. Practice integrating multiple ML/NLP libraries
5. Develop skills in system architecture and scalability

## Notes

- The current implementation prioritizes **simplicity** and **learning** over performance
- All code is intentionally straightforward to encourage understanding and modification
- Start by running the system as-is, then gradually add features
- Document your changes and learnings as you enhance the system

---

*Template designed for hands-on learning and experimentation with RAG systems*