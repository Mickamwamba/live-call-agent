# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from docProcessor import DocumentProcessor
import os
import time
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentAssistant: 
    def __init__(self,pdf_directory):
        """Initialize the LLM-based code review agent"""
        self.pdf_directory = pdf_directory
        self.doc_processor = DocumentProcessor(pdf_directory)
        self.vectorstore = None
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.initialized = False
        pass
    
    async def initialize(self):
        """Initialize the agent by processing documents"""
        if not self.initialized:
            logger.info("Initializing code review agent")
            self.vectorstore = await self.doc_processor.process_documents()
            self.initialized = True
            logger.info("Code review agent initialized successfully")

    async def get_resolutions(self, query) -> str:
        """Return resolutions based on the internal documents"""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()

        logger.info(f"Starting Fetching Resolutions")

        #step1: Retrieve relevant resolutions from the vector store: 
        docs = self.vectorstore.similarity_search(query, k=3)

        # Step 2: Format document context
        standards_context = "\n\n".join([
            f"Document: {doc.metadata['source']} (Page {doc.metadata['page']})\n{doc.page_content}"
            for doc in docs
        ])

        prompt_template = """
        You are a customer service manager. Based on the customer issue below, provide: 1) Recommended procedures (array of strings) 2) 
        Mock previous tickets (array with id, date, issue, status) 3) Customer mood assessment. Return as JSON with fields: procedures, previous_tickets, customer_mood

        Customer Issue: {case_info}
        Below are relevant excerpts from our **internal company procedures, troubleshooting guides, and policy documents**:

        {standards_context}
            """

        prompt = PromptTemplate(
            template=prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        logger.info("Reviewing...")
        review = await chain.arun(
            standards_context=standards_context,
            case_info = query
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Review completed in {elapsed_time:.2f} seconds")

        return review


    async def add_document(self, file_path: str) -> bool:
        """Add a new document to the knowledge base"""
        success = await self.doc_processor.add_document(file_path)
        if success:
            self.initialized = False  # Force reprocess
        return success
    