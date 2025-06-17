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
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
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

class CodeReviewAgent:
    def __init__(self, pdf_directory: str):
        """Initialize the LLM-based code review agent"""
        self.pdf_directory = pdf_directory
        self.doc_processor = DocumentProcessor(pdf_directory)
        self.vectorstore = None
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent by processing documents"""
        if not self.initialized:
            logger.info("Initializing code review agent")
            self.vectorstore = await self.doc_processor.process_documents()
            self.initialized = True
            logger.info("Code review agent initialized successfully")
    
    async def review_code(self, code: str, language: str = "python", file_name: str = None) -> str:
        """Review code against standards using LLM reasoning"""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        logger.info(f"Starting code review for {language} code")
        
        # Step 1: Retrieve relevant standards from the vector store
        queries = [
            f"{language} coding standards",
            f"{language} best practices",
            f"DO-178 requirements for {language}",
            f"{language} common errors and issues"
        ]
        
        if file_name:
            # Add more specific query if filename is provided
            queries.append(f"standards for {file_name} or similar files")
        
        all_relevant_docs = []
        for query in queries:
            docs = self.vectorstore.similarity_search(query, k=3)
            all_relevant_docs.extend(docs)
        
        # Remove duplicate documents
        unique_docs = []
        seen_texts = set()
        for doc in all_relevant_docs:
            if doc.page_content not in seen_texts:
                unique_docs.append(doc)
                seen_texts.add(doc.page_content)
        
        logger.info(f"Retrieved {len(unique_docs)} relevant document chunks")
        
        # Format document content for context
        standards_context = "\n\n".join([
            f"Document: {doc.metadata['source']} (Page {doc.metadata['page']})\n{doc.page_content}"
            for doc in unique_docs
        ])
        
        # Step 2: Create a prompt for the LLM to review the code
        prompt_template = """
        You are an expert code reviewer specializing in compliance with coding standards and safety-critical software requirements. 
        
        Below are relevant excerpts from coding standards, best practices, and DO-178 requirements:
        
        {standards_context}
        
        Please review the following {language} code{file_info} against these standards:
        
        ```{language}
        {code}
        ```
        
        Review instructions:
        1. Analyze the code for compliance with the standards and best practices mentioned above
        2. Identify any violations or deviations from the standards
        3. For each issue found, provide:
           - A description of what's wrong
           - The specific standard or best practice that's violated
           - A practical suggestion with example code showing how to fix it
           - If relevant, explain the DO-178 compliance implications
        4. Rate the severity of each issue (Critical, High, Medium, Low)
        5. If the code complies with all standards, note that as well
        
        Format your response as a structured code review with clear sections. Start with a summary of findings.
        Use markdown formatting to make the review readable.
        """
        
        file_info = f" in file '{file_name}'" if file_name else ""
        
        prompt = PromptTemplate(
            input_variables=["standards_context", "language", "code", "file_info"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Step 3: Generate the code review
        logger.info("Generating code review")
        review = await chain.arun(
            standards_context=standards_context,
            language=language,
            code=code,
            file_info=file_info
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Code review completed in {elapsed_time:.2f} seconds")
        
        return review
    
    async def add_document(self, file_path: str) -> bool:
        """Add a new document to the knowledge base"""
        success = await self.doc_processor.add_document(file_path)
        if success:
            # Reset initialization to force reprocessing documents
            self.initialized = False
        return success

# FastAPI application
app = FastAPI(title="LLM Code Review Agent")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class CodeReviewRequest(BaseModel):
    code: str
    language: str = "python"
    file_name: str = None

# Initialize the agent
pdf_directory = os.getenv("PDF_DIRECTORY", "./documents")
agent = CodeReviewAgent(pdf_directory)

@app.on_event("startup")
async def startup_event():
    # Initialize the agent in the background
    asyncio.create_task(agent.initialize())

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/review")
async def review_code(request: Request):
    """Review code and provide feedback"""
    # Parse the JSON request body manually
    try:
        data = await request.json()
        code = data.get("code", "")
        language = data.get("language", "python")
        file_name = data.get("file_name", None)
        
        # Validate required fields
        if not code:
            return {"error": "Code field is required"}
        
        feedback = await agent.review_code(
            code, 
            language,
            file_name
        )
        return {"feedback": feedback}
    except Exception as e:
        logger.error(f"Error in review_code: {e}")
        return {"error": str(e), "feedback": "An error occurred during code review"}
    

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new standards document"""
    try:
        # Save the uploaded file
        file_path = f"./uploads/{file.filename}"
        os.makedirs("./uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Add the document to the agent
        success = await agent.add_document(file_path)
        
        if success:
            return {"message": f"Document {file.filename} uploaded and processed successfully"}
        else:
            return {"message": f"Error processing document {file.filename}"}
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return {"error": str(e)}

# Create HTML templates
def create_templates():
    """Create necessary HTML templates"""
    os.makedirs("templates", exist_ok=True)
    
    # Create index.html
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Code Review Agent</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
        <link rel="stylesheet" href="/static/styles.css">
    </head>
    <body>
        <div class="container mt-5">
            <h1 class="mb-4">LLM Code Review Agent</h1>
            
            <ul class="nav nav-tabs" id="myTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="review-tab" data-bs-toggle="tab" data-bs-target="#review" type="button" role="tab">Code Review</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="upload-tab" data-bs-toggle="tab" data-bs-target="#upload" type="button" role="tab">Upload Standards</button>
                </li>
            </ul>
            
            <div class="tab-content mt-3" id="myTabContent">
                <!-- Code Review Tab -->
                <div class="tab-pane fade show active" id="review" role="tabpanel">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="form-group mb-3">
                                <label for="language">Language:</label>
                                <select class="form-control" id="language">
                                    <option value="python">Python</option>
                                    <option value="c">C</option>
                                    <option value="cpp">C++</option>
                                    <option value="java">Java</option>
                                    <option value="javascript">JavaScript</option>
                                </select>
                            </div>
                            
                            <div class="form-group mb-3">
                                <label for="filename">File name (optional):</label>
                                <input type="text" class="form-control" id="filename" placeholder="e.g. controller.py">
                            </div>
                            
                            <div class="form-group mb-3">
                                <label for="code">Code to review:</label>
                                <textarea class="form-control" id="code" rows="15" placeholder="Paste your code here..."></textarea>
                            </div>
                            
                            <button class="btn btn-primary" id="reviewButton">Review Code</button>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Review Results</h5>
                                </div>
                                <div class="card-body">
                                    <div id="results">
                                        <div class="text-center text-muted">
                                            <p>Review results will appear here</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Upload Tab -->
                <div class="tab-pane fade" id="upload" role="tabpanel">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Upload Standards Document</h5>
                                </div>
                                <div class="card-body">
                                    <form id="uploadForm" enctype="multipart/form-data">
                                        <div class="form-group mb-3">
                                            <label for="documentFile">Select PDF document:</label>
                                            <input type="file" class="form-control" id="documentFile" accept=".pdf" required>
                                        </div>
                                        <button type="submit" class="btn btn-primary">Upload Document</button>
                                    </form>
                                    <div id="uploadStatus" class="mt-3"></div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Tips for Documents</h5>
                                </div>
                                <div class="card-body">
                                    <ul>
                                        <li>Upload all your coding standards as PDF files</li>
                                        <li>Include document names like "DO-178.pdf" or "coding_standards.pdf" to help categorization</li>
                                        <li>Each document will be processed and added to the knowledge base</li>
                                        <li>The agent will automatically consider all uploaded documents when reviewing code</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js"></script>
        <script src="/static/script.js"></script>
    </body>
    </html>
    """
    
    with open("templates/index.html", "w") as f:
        f.write(index_html)
    
    # Create static folder and CSS file
    os.makedirs("static", exist_ok=True)
    
    styles_css = """
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0, 0, 0, 0.3);
        border-radius: 50%;
        border-top-color: #007bff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    pre {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 4px;
    }
    
    .severity-critical {
        color: #dc3545;
        font-weight: bold;
    }
    
    .severity-high {
        color: #fd7e14;
        font-weight: bold;
    }
    
    .severity-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .severity-low {
        color: #6c757d;
    }
    """
    
    with open("static/styles.css", "w") as f:
        f.write(styles_css)
    
    # Create JavaScript file
    script_js = """
    document.addEventListener('DOMContentLoaded', function() {
    const reviewButton = document.getElementById('reviewButton');
    const codeTextarea = document.getElementById('code');
    const languageSelect = document.getElementById('language');
    const filenameInput = document.getElementById('filename');
    const resultsDiv = document.getElementById('results');
    const uploadForm = document.getElementById('uploadForm');
    const uploadStatus = document.getElementById('uploadStatus');
    
    // Handle code review submission
    reviewButton.addEventListener('click', async function() {
        const code = codeTextarea.value.trim();
        if (!code) {
            alert('Please enter some code to review');
            return;
        }
        
        // Show loading indicator
        resultsDiv.innerHTML = '<div class="text-center"><div class="loading"></div><p class="mt-2">Analyzing code...</p></div>';
        
        try {
            const response = await fetch('/review', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: code,
                    language: languageSelect.value,
                    file_name: filenameInput.value || null
                }),
            });
            
            // Check if the response is ok
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server returned ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            
            // Check if data.feedback exists before using it
            if (data && data.feedback) {
                // Format the feedback with markdown
                resultsDiv.innerHTML = marked.parse(data.feedback);
                
                // Apply syntax highlighting to code blocks
                document.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightBlock(block);
                });
                
                // Add severity classes
                addSeverityClasses();
            } else {
                resultsDiv.innerHTML = '<div class="alert alert-warning">No feedback data received from server</div>';
                console.error('Received data:', data);
            }
            
        } catch (error) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            console.error('Error during review:', error);
        }
    });
    
    // Handle document upload
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('documentFile');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a PDF file to upload');
            return;
        }
        
        // Show loading
        uploadStatus.innerHTML = '<div class="alert alert-info">Uploading and processing document...</div>';
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server returned ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                uploadStatus.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
            } else {
                uploadStatus.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
                fileInput.value = '';
            }
            
        } catch (error) {
            uploadStatus.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            console.error('Error during upload:', error);
        }
    });
    
    // Function to add severity classes for styling
    function addSeverityClasses() {
        const content = resultsDiv.innerHTML;
        let updatedContent = content;
        
        // Add classes for severity levels
        updatedContent = updatedContent.replace(/Severity: Critical/g, '<span class="severity-critical">Severity: Critical</span>');
        updatedContent = updatedContent.replace(/Severity: High/g, '<span class="severity-high">Severity: High</span>');
        updatedContent = updatedContent.replace(/Severity: Medium/g, '<span class="severity-medium">Severity: Medium</span>');
        updatedContent = updatedContent.replace(/Severity: Low/g, '<span class="severity-low">Severity: Low</span>');
        
        resultsDiv.innerHTML = updatedContent;
    }
});
    """
    
    with open("static/script.js", "w") as f:
        f.write(script_js)

if __name__ == "__main__":
    # Create templates
    create_templates()
    
    # Start the application
    uvicorn.run(app, host="0.0.0.0", port=8000)
