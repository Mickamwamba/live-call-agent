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

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes and origins

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1
CHUNK_SIZE = 1024 * 8
REGION = "us-east-1"

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
                    await self.queue.put(f"data: {alt.transcript}\n\n")

@app.route('/transcribe', methods=['GET'])
def transcribe():
    # audio_file = request.files.get('file')
    # if not audio_file:
    #     return "No file provided", 400

    # filename = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    # audio_file.save(filename)
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
                # Wrap your data in JSON
                json_data = json.dumps({
                    "transcript": data,
                    # optionally add: chunk_number, confidence, timestamp, etc.
                })
                # yield data
                yield f"data: {json_data}\n\n"
            except Exception as e:
                yield f"data: Error - {str(e)}\n\n"
                break
        loop.close()

    return Response(stream_with_context(stream_response()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
