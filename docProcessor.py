import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
import time
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import logging
from pathlib import Path
import asyncio

# Import necessary LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
        self.embeddings = OpenAIEmbeddings()
        # Create directory if it doesn't exist
        Path(pdf_directory).mkdir(parents=True, exist_ok=True)
    
    async def process_documents(self) -> Chroma:
        """Process all PDFs in the directory and create a vector store"""
        logger.info(f"Processing documents from {self.pdf_directory}")
        documents = []
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(self.pdf_directory, pdf_file)
            logger.info(f"Processing {file_path}")
            
            # Allow other tasks to run
            await asyncio.sleep(0)
            
            doc = fitz.open(file_path)
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Add metadata about document
                metadata = {
                    "source": pdf_file,
                    "page": page_num + 1,
                    "type": self._determine_document_type(pdf_file)
                }
                
                documents.append({"text": text, "metadata": metadata})
        
        # Split texts into chunks
        logger.info("Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        processed_documents = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["text"])
            for chunk in chunks:
                processed_documents.append({
                    "text": chunk, 
                    "metadata": doc["metadata"]
                })
        
        logger.info(f"Created {len(processed_documents)} document chunks")
        
        # Create vector store
        logger.info("Creating vector embeddings")
        vectorstore = Chroma.from_texts(
            [doc["text"] for doc in processed_documents],
            self.embeddings,
            metadatas=[doc["metadata"] for doc in processed_documents]
        )
        
        logger.info("Vector store created successfully")
        return vectorstore
    
    def _determine_document_type(self, filename: str) -> str:
        """Determine document type based on filename"""
        filename_lower = filename.lower()
        if "do-178" in filename_lower or "do178" in filename_lower:
            return "do178"
        elif "standard" in filename_lower or "guideline" in filename_lower:
            return "standard"
        elif "cookbook" in filename_lower or "recipe" in filename_lower:
            return "cookbook"
        elif "best" in filename_lower and "practice" in filename_lower:
            return "best_practice"
        else:
            return "general"
            
    async def add_document(self, file_path: str) -> bool:
        """Add a new document to the directory"""
        try:
            # Copy file to document directory
            destination = os.path.join(self.pdf_directory, os.path.basename(file_path))
            os.rename(file_path, destination)
            logger.info(f"Added document: {destination}")
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
