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
        print("transcript===============>>",transcript)
        
        if not OPENAI_API_KEY:
            return {"error": "OpenAI API key not configured"}, 500
        
        # Call OpenAI API
        response = requests.post(
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
                        'content': f'Please analyze this customer call transcript: {transcript}'
                    }
                ],
                'response_format': {'type': 'json_object'}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extract the content from OpenAI response
            summary_content = result['choices'][0]['message']['content']
            print("xxxxxxxxxxxxxxxxxxxxxxxxx",summary_content)
            # Parse the JSON response from OpenAI
            try:
                summary_data = json.loads(summary_content)
                return summary_data, 200
            except json.JSONDecodeError:
                # If OpenAI didn't return valid JSON, wrap it in a basic structure
                return {
                    "summary": summary_content,
                    "key_issues": [],
                    "customer_mood": {"tone": "unknown", "confidence": 0}
                }, 200
        else:
            return {"error": f"OpenAI API error: {response.status_code}"}, 500
            
    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}, 500
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)