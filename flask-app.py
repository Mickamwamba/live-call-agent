# import asyncio
# import aiofile
# import os
# from flask import Flask, request, Response, stream_with_context
# from amazon_transcribe.client import TranscribeStreamingClient
# from amazon_transcribe.handlers import TranscriptResultStreamHandler
# from amazon_transcribe.model import TranscriptEvent
# from amazon_transcribe.utils import apply_realtime_delay
# from flask_cors import CORS
# import json

# app = Flask(__name__)
# CORS(app)  # Enables CORS for all routes and origins

# SAMPLE_RATE = 16000
# BYTES_PER_SAMPLE = 2
# CHANNEL_NUMS = 1
# CHUNK_SIZE = 1024 * 8
# REGION = "us-east-1"

# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# class SSETranscriptHandler(TranscriptResultStreamHandler):
#     def __init__(self, output_stream, queue):
#         super().__init__(output_stream)
#         self.queue = queue

#     async def handle_transcript_event(self, transcript_event: TranscriptEvent):
#         for result in transcript_event.transcript.results:
#             for alt in result.alternatives:
#                 if alt.transcript:
#                     # Put the transcript and isPartial as a dictionary
#                     transcript_data = {
#                         "transcript": alt.transcript,
#                         "isPartial": result.is_partial
#                     }
#                     await self.queue.put(transcript_data)

# @app.route('/transcribe', methods=['GET'])
# def transcribe():
#     filename = "call.wav"
    
#     queue = asyncio.Queue()

#     async def transcribe_audio():
#         client = TranscribeStreamingClient(region=REGION)
#         stream = await client.start_stream_transcription(
#             language_code="en-US",
#             media_sample_rate_hz=SAMPLE_RATE,
#             media_encoding="pcm",
#         )

#         async def write_chunks():
#             async with aiofile.AIOFile(filename, "rb") as afp:
#                 reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
#                 await apply_realtime_delay(
#                     stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
#                 )
#             await stream.input_stream.end_stream()

#         handler = SSETranscriptHandler(stream.output_stream, queue)
#         await asyncio.gather(write_chunks(), handler.handle_events())
#         await queue.put(None)  # sentinel for end of stream

#     def stream_response():
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         loop.create_task(transcribe_audio())

#         while True:
#             try:
#                 data = loop.run_until_complete(queue.get())
#                 if data is None:
#                     break
                
#                 # Data is now already a dictionary with transcript and isPartial
#                 json_data = json.dumps(data)
#                 yield f"data: {json_data}\n\n"
                
#             except Exception as e:
#                 error_data = {
#                     "transcript": f"Error - {str(e)}",
#                     "isPartial": False
#                 }
#                 yield f"data: {json.dumps(error_data)}\n\n"
#                 break
#         loop.close()

#     return Response(stream_with_context(stream_response()), mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True, threaded=True)

import asyncio
import aiofile
import os
from flask import Flask, request, Response, stream_with_context
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from amazon_transcribe.utils import apply_realtime_delay
from flask_cors import CORS
import json
import requests

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

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from agent2 import CodeReviewAgent
from agent_main import AgentAssistant
import asyncio
from langchain_community.embeddings import OpenAIEmbeddings

app = Flask(__name__)
from dotenv import load_dotenv

CORS(app)  # Enables CORS for all routes and origins

# Load environment variables from .env file
load_dotenv()

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1
CHUNK_SIZE = 1024 * 8
REGION = "us-east-1"

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pdf_directory = os.getenv("PDF_DIRECTORY", "./documents")
# agent = CodeReviewAgent(pdf_directory)

agent = AgentAssistant(pdf_directory)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSETranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, queue):
        super().__init__(output_stream)
        self.queue = queue

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            for alt in result.alternatives:
                if alt.transcript:
                    # Put the transcript and isPartial as a dictionary
                    transcript_data = {
                        "transcript": alt.transcript,
                        "isPartial": result.is_partial
                    }
                    await self.queue.put(transcript_data)

@app.route('/transcribe', methods=['GET'])
def transcribe():
    filename = "call.wav"
    
    queue = asyncio.Queue()

    async def transcribe_audio():
        client = TranscribeStreamingClient(region=REGION)
        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
        )

        async def write_chunks():
            async with aiofile.AIOFile(filename, "rb") as afp:
                reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
                await apply_realtime_delay(
                    stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
                )
            await stream.input_stream.end_stream()

        handler = SSETranscriptHandler(stream.output_stream, queue)
        await asyncio.gather(write_chunks(), handler.handle_events())
        await queue.put(None)  # sentinel for end of stream

    def stream_response():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(transcribe_audio())

        while True:
            try:
                data = loop.run_until_complete(queue.get())
                if data is None:
                    break
                
                # Data is now already a dictionary with transcript and isPartial
                json_data = json.dumps(data)
                yield f"data: {json_data}\n\n"
                
            except Exception as e:
                error_data = {
                    "transcript": f"Error - {str(e)}",
                    "isPartial": False
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break
        loop.close()

    return Response(stream_with_context(stream_response()), mimetype='text/event-stream')

@app.route('/summarize', methods=['POST'])
def summarize_call():
    try:
        # Get the transcript from the request
        data = request.get_json()
        if not data or 'transcript' not in data:
            return {"error": "Transcript is required"}, 400
        
        transcript = data['transcript']
        
        if not OPENAI_API_KEY:
            return {"error": "OpenAI API key not configured"}, 500
        
        # Generate summary and analysis
        summary_response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-4o-mini',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a customer service AI agent. Analyze the customer transcription and provide: 1) A brief summary (2-3 sentences) 2) List of key issues (3-5 items) 3) Customer mood analysis with tone and confidence percentage. Return as JSON with fields: summary, key_issues (array), customer_mood (object with tone and confidence).'
                    },
                    {
                        'role': 'user',
                        'content': transcript
                    }
                ],
                'response_format': {'type': 'json_object'}
            },
            timeout=30
        )
        
        if summary_response.status_code == 200:
            summary_result = summary_response.json()
            summary_data = json.loads(summary_result['choices'][0]['message']['content'])
            
            # Ensure default values
            result = {
                'summary': summary_data.get('summary', 'Customer call processed'),
                'key_issues': summary_data.get('key_issues', ['Technical inquiry']),
                'customer_mood': summary_data.get('customer_mood', {'tone': 'Neutral', 'confidence': 75})
            }
            return result, 200
        else:
            return {"error": f"OpenAI API error: {summary_response.status_code}"}, 500
            
    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}, 500
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500

@app.route('/resolutions', methods=['POST'])
def get_resolutions():
    try:
        # Get the transcript from the request
        data = request.get_json()
        if not data or 'transcript' not in data:
            return {"error": "Transcript is required"}, 400
        
        transcript = data['transcript']
        
        if not OPENAI_API_KEY:
            return {"error": "OpenAI API key not configured"}, 500
        
        # Generate resolution suggestions
        resolution_response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-4o-mini',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a technical support agent. Based on the customer issue, provide 2-3 potential resolutions. For each resolution, provide: title, source, confidence (0-100), steps (array of strings), category. Return as JSON with "resolutions" array field.'
                    },
                    {
                        'role': 'user',
                        'content': f'Customer issue: {transcript}'
                    }
                ],
                'response_format': {'type': 'json_object'}
            },
            timeout=30
        )
        
        if resolution_response.status_code == 200:
            resolution_result = resolution_response.json()
            resolution_data = json.loads(resolution_result['choices'][0]['message']['content'])
            
            result = {
                'resolutions': resolution_data.get('resolutions', [])
            }
            return result, 200
        else:
            return {"error": f"OpenAI API error: {resolution_response.status_code}"}, 500
            
    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}, 500
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500

@app.route('/test', methods=['GET'])
def get_insights2():
    # initialize the agent which is connected to the vector store: - it was initialized before already 
    query = "The customer is calling Bright Connect customer support to inquire about an unrecognized charge on their recent bill. They seek assistance in understanding the billing details."
    res = asyncio.run(agent.get_resolutions(query))
    print(res)
    result = {"data": res}
    return result,200




@app.route('/insights', methods=['POST'])
def get_insights():
    try:
        # Get the transcript from the request
        data = request.get_json()
        if not data or 'transcript' not in data:
            return {"error": "Transcript is required"}, 400
        
        transcript = data['transcript']
        
        if not OPENAI_API_KEY:
            return {"error": "OpenAI API key not configured"}, 500
        
        # Generate insights and procedures
        insights_response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-4o-mini',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a customer service manager. Based on the customer issue, provide: 1) Recommended procedures (array of strings) 2) Mock previous tickets (array with id, date, issue, status) 3) Customer mood assessment. Return as JSON with fields: procedures, previous_tickets, customer_mood.'
                    },
                    {
                        'role': 'user',
                        'content': f'Customer issue: {transcript}'
                    }
                ],
                'response_format': {'type': 'json_object'}
            },
            timeout=30
        )
        
        if insights_response.status_code == 200:
            insights_result = insights_response.json()
            insights_data = json.loads(insights_result['choices'][0]['message']['content'])
            
            result = {
                'procedures': insights_data.get('procedures', ['Follow standard procedures']),
                'previous_tickets': insights_data.get('previous_tickets', []),
                'customer_mood': insights_data.get('customer_mood', {'tone': 'Neutral', 'confidence': 75})
            }
            return result, 200
        else:
            return {"error": f"OpenAI API error: {insights_response.status_code}"}, 500
            
    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}, 500
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new standards document"""
    # print(f"Received file: {file}, type: {type(file)}")


    # print("file================>",file)
    try:
        # Save the uploaded file
        # file_path = f"./uploads/{file.filename}"
        # os.makedirs("./uploads", exist_ok=True)
        file_path = "/Users/michaelkimollo/DSProjects/impetus-hackathon/documents/do_178b.pdf"
        # with open(file_path, "wb") as f:
        #     content = await file.read()
        #     f.write(content)
        
        # Add the document to the agent
        success = await agent.add_document(file_path)
        
        if success:
            # return {"message": f"Document {file.filename} uploaded and processed successfully"}
            return {"message": f"Document uploaded and processed successfully"}
        
        else:
            return {"message": f"Error processing document {file.filename}"}
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # res = asyncio.run(agent.review_code("The customer is calling Bright Connect customer support to inquire about an unrecognized charge on their recent bill. They seek assistance in understanding the billing details."))
    # review = await agent.review_code(code="your code here", language="python")

    app.run(debug=True, threaded=True)
