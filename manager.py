# import time
# from audio_processor import RealTimeAudioProcessor
# from transcriber_ import AWSTranscribeClient

# class LiveTranscriptionManager:
#     def __init__(self, file_path, region='us-east-1', language='en-US', chunk_duration=30.0):
#         self.processor = RealTimeAudioProcessor(file_path, chunk_duration)
#         self.transcriber = AWSTranscribeClient(region, language)
#         self.chunk_duration = chunk_duration

#     def run(self):
#         info = self.processor.get_audio_info()
#         duration = info['duration']
#         results = []

#         try:
#             for i in range(int(duration // self.chunk_duration) + 1):
#                 start = i * self.chunk_duration
#                 chunk = self.processor.extract_chunk(start)
#                 if not chunk:
#                     break

#                 key = f"chunk_{i}_{int(time.time())}.wav"
#                 uri = self.transcriber.upload_audio(chunk['file_path'], key)
#                 job_name = f"job_{i}_{int(time.time())}"
#                 self.transcriber.start_job(job_name, uri)
#                 job = self.transcriber.wait_for_completion(job_name)
#                 result = self.transcriber.get_result(job)
#                 self.transcriber.cleanup(key)
#                 results.append({
#                     'chunk': i,
#                     'start': chunk['start_time'],
#                     'text': result['transcript'],
#                     'segments': result['segments']
#                 })
#         finally:
#             self.transcriber.delete_bucket()

#         return results

# manager.py
import time
import json
from audio_processor import RealTimeAudioProcessor
from transcriber_ import AWSTranscribeClient

class LiveTranscriptionStreamer:
    def __init__(self, file_path, region='us-east-1', language='en-US', chunk_duration=30.0):
        self.processor = RealTimeAudioProcessor(file_path, chunk_duration)
        self.transcriber = AWSTranscribeClient(region, language)
        self.chunk_duration = chunk_duration

    def stream_transcriptions(self):
        info = self.processor.get_audio_info()
        duration = info['duration']

        try:
            for i in range(int(duration // self.chunk_duration) + 1):
                start = i * self.chunk_duration
                chunk = self.processor.extract_chunk(start)
                if not chunk:
                    break

                key = f"chunk_{i}_{int(time.time())}.wav"
                uri = self.transcriber.upload_audio(chunk['file_path'], key)
                job_name = f"job_{i}_{int(time.time())}"
                self.transcriber.start_job(job_name, uri)
                job = self.transcriber.wait_for_completion(job_name)
                result = self.transcriber.get_result(job)
                self.transcriber.cleanup(key)

                yield json.dumps({
                    'chunk': i,
                    'start': chunk['start_time'],
                    'text': result['transcript'],
                    'segments': result['segments']
                }) + "\n"
        finally:
            self.transcriber.delete_bucket()
