import asyncio
import boto3
import wave
import time
import tempfile
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
import json

class RealTimeAudioProcessor:
    def __init__(self, audio_file_path, chunk_duration=30.0):
        self.audio_file_path = audio_file_path
        self.chunk_duration = chunk_duration  # 30 seconds
        self.audio_info = None
        self.current_position = 0
        
    def get_audio_info(self):
        """Get comprehensive audio file information"""
        try:
            with wave.open(self.audio_file_path, 'rb') as wav_file:
                self.audio_info = {
                    'sample_rate': wav_file.getframerate(),
                    'channels': wav_file.getnchannels(),
                    'sample_width': wav_file.getsampwidth(),
                    'total_frames': wav_file.getnframes(),
                    'duration': wav_file.getnframes() / wav_file.getframerate(),
                    'bits_per_sample': wav_file.getsampwidth() * 8
                }
                return self.audio_info
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None
    
    def extract_chunk(self, start_time, duration):
        """Extract a specific time chunk from the audio file"""
        try:
            with wave.open(self.audio_file_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Calculate frame positions
                start_frame = int(start_time * sample_rate)
                frames_to_read = int(duration * sample_rate)
                
                # Set position and read frames
                wav_file.setpos(start_frame)
                audio_data = wav_file.readframes(frames_to_read)
                
                if not audio_data:
                    return None
                
                # Create temporary file for this chunk
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                
                with wave.open(temp_file.name, 'wb') as chunk_wav:
                    chunk_wav.setnchannels(channels)
                    chunk_wav.setsampwidth(sample_width)
                    chunk_wav.setframerate(sample_rate)
                    chunk_wav.writeframes(audio_data)
                
                return {
                    'file_path': temp_file.name,
                    'start_time': start_time,
                    'duration': len(audio_data) / (sample_rate * channels * sample_width),
                    'actual_frames': len(audio_data) // (channels * sample_width)
                }
                
        except Exception as e:
            print(f"Error extracting chunk at {start_time}s: {e}")
            return None

class AWSTranscribeRealTime:
    def __init__(self, region='us-east-1', language_code='en-US'):
        self.region = region
        self.language_code = language_code
        self.transcribe_client = boto3.client('transcribe', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Create a unique S3 bucket name for temporary files
        self.bucket_name = f"transcribe-temp-{uuid.uuid4().hex[:8]}"
        self.setup_s3_bucket()
        
    def setup_s3_bucket(self):
        """Create S3 bucket for temporary audio files"""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            print(f"Created S3 bucket: {self.bucket_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyExists':
                print(f"S3 bucket {self.bucket_name} already exists")
            else:
                print(f"Error creating S3 bucket: {e}")
                raise
    
    def upload_audio_to_s3(self, local_file_path, s3_key):
        """Upload audio file to S3"""
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"Uploaded to S3: {s3_uri}")
            return s3_uri
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            return None
    
    def start_transcription_job(self, s3_uri, job_name):
        """Start AWS Transcribe job"""
        try:
            response = self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': s3_uri},
                MediaFormat='wav',
                LanguageCode=self.language_code,
                Settings={
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': 2,  # Customer and Agent
                    'ShowAlternatives': True,
                    'MaxAlternatives': 2
                }
            )
            return response['TranscriptionJob']
        except Exception as e:
            print(f"Error starting transcription job {job_name}: {e}")
            return None
    
    def wait_for_transcription_completion(self, job_name, timeout=60):
        """Wait for transcription job to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                
                job = response['TranscriptionJob']
                status = job['TranscriptionJobStatus']
                
                if status == 'COMPLETED':
                    return job
                elif status == 'FAILED':
                    print(f"Transcription job {job_name} failed: {job.get('FailureReason', 'Unknown error')}")
                    return None
                
                # Wait before checking again
                # await asyncio.sleep(2)
                asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error checking job status: {e}")
                return None
        
        print(f"Transcription job {job_name} timed out after {timeout} seconds")
        return None
    
    async def wait_for_transcription_completion_async(self, job_name, timeout=60):
        """Async version of wait_for_transcription_completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                
                job = response['TranscriptionJob']
                status = job['TranscriptionJobStatus']
                
                if status == 'COMPLETED':
                    return job
                elif status == 'FAILED':
                    print(f"Transcription job {job_name} failed: {job.get('FailureReason', 'Unknown error')}")
                    return None
                
                # Wait before checking again
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error checking job status: {e}")
                return None
        
        print(f"Transcription job {job_name} timed out after {timeout} seconds")
        return None
    
    def get_transcription_result(self, job):
        """Extract transcription text from completed job"""
        try:
            transcript_uri = job['Transcript']['TranscriptFileUri']
            
            # Download transcript from S3
            import urllib.request
            with urllib.request.urlopen(transcript_uri) as response:
                transcript_data = json.loads(response.read().decode())
            
            # Extract transcript and confidence
            results = transcript_data['results']
            transcript_text = results['transcripts'][0]['transcript']
            
            # Calculate average confidence
            items = results.get('items', [])
            confidences = [float(item.get('alternatives', [{}])[0].get('confidence', 0)) 
                          for item in items if item.get('alternatives')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Extract speaker labels if available
            speaker_labels = results.get('speaker_labels', {})
            segments = speaker_labels.get('segments', [])
            
            return {
                'transcript': transcript_text,
                'confidence': avg_confidence,
                'speaker_segments': segments,
                'duration': job.get('MediaSampleRateHertz', 0),
                'language_code': job.get('LanguageCode', self.language_code)
            }
            
        except Exception as e:
            print(f"Error extracting transcription result: {e}")
            return None
    
    def cleanup_s3_objects(self, s3_key):
        """Clean up S3 objects"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
        except Exception as e:
            print(f"Error cleaning up S3 object {s3_key}: {e}")
    
    def cleanup_s3_bucket(self):
        """Clean up the entire S3 bucket"""
        try:
            # Delete all objects first
            objects = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in objects:
                for obj in objects['Contents']:
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
            
            # Delete the bucket
            self.s3_client.delete_bucket(Bucket=self.bucket_name)
            print(f"Cleaned up S3 bucket: {self.bucket_name}")
        except Exception as e:
            print(f"Error cleaning up S3 bucket: {e}")

class LiveTranscriptionManager:
    def __init__(self, audio_file_path, aws_region='us-east-1', language_code='en-US', chunk_duration=30.0):
        self.audio_file_path = audio_file_path
        self.chunk_duration = chunk_duration
        self.audio_processor = RealTimeAudioProcessor(audio_file_path, chunk_duration)
        self.transcribe_client = AWSTranscribeRealTime(aws_region, language_code)
        self.transcription_results = []
        self.callbacks = []
        self.is_running = False
        
    def add_callback(self, callback_func):
        """Add callback function for transcription results"""
        self.callbacks.append(callback_func)
        
    async def call_callbacks(self, result):
        """Call all registered callbacks"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                print(f"Error in callback: {e}")
    
    async def process_chunk(self, chunk_info, chunk_number):
        """Process a single audio chunk through AWS Transcribe"""
        try:
            print(f"\n{'='*50}")
            print(f"PROCESSING CHUNK {chunk_number}")
            print(f"Time: {chunk_info['start_time']:.1f}s - {chunk_info['start_time'] + chunk_info['duration']:.1f}s")
            print(f"Duration: {chunk_info['duration']:.1f}s")
            print(f"{'='*50}")
            
            # Upload to S3
            s3_key = f"chunk_{chunk_number}_{int(time.time())}.wav"
            s3_uri = self.transcribe_client.upload_audio_to_s3(chunk_info['file_path'], s3_key)
            
            if not s3_uri:
                return None
            
            # Start transcription job
            job_name = f"transcribe_chunk_{chunk_number}_{int(time.time())}"
            print(f"Starting transcription job: {job_name}")
            
            job_info = self.transcribe_client.start_transcription_job(s3_uri, job_name)
            if not job_info:
                return None
            
            # Wait for completion
            print(f"Waiting for transcription to complete...")
            completed_job = await self.transcribe_client.wait_for_transcription_completion_async(job_name)
            
            if completed_job:
                # Get results
                transcript_result = self.transcribe_client.get_transcription_result(completed_job)
                
                if transcript_result:
                    result = {
                        'chunk_number': chunk_number,
                        'start_time': chunk_info['start_time'],
                        'duration': chunk_info['duration'],
                        'transcript': transcript_result['transcript'],
                        'confidence': transcript_result['confidence'],
                        'speaker_segments': transcript_result['speaker_segments'],
                        'timestamp': time.time(),
                        'job_name': job_name
                    }
                    
                    self.transcription_results.append(result)
                    
                    print(f"✅ TRANSCRIPTION RESULT:")
                    print(f"   Text: {result['transcript']}")
                    print(f"   Confidence: {result['confidence']:.2f}")
                    
                    # Call callbacks
                    await self.call_callbacks(result)
                    
                    # Cleanup
                    self.transcribe_client.cleanup_s3_objects(s3_key)
                    
                    return result
            
            return None
            
        except Exception as e:
            print(f"Error processing chunk {chunk_number}: {e}")
            return None
        finally:
            # Always cleanup temp file
            if os.path.exists(chunk_info['file_path']):
                os.unlink(chunk_info['file_path'])
    
    async def start_live_transcription(self):
        """Start real-time transcription with 30-second chunks"""
        print(f"🎙️  Starting REAL AWS Transcribe - 30 Second Chunks")
        print(f"Audio File: {self.audio_file_path}")
        
        # Get audio info
        audio_info = self.audio_processor.get_audio_info()
        if not audio_info:
            print("❌ Failed to read audio file")
            return []
        
        print(f"📊 Audio Info:")
        print(f"   Duration: {audio_info['duration']:.1f} seconds")
        print(f"   Sample Rate: {audio_info['sample_rate']} Hz")
        print(f"   Channels: {audio_info['channels']}")
        print(f"   Bits per Sample: {audio_info['bits_per_sample']}")
        
        total_chunks = int(audio_info['duration'] / self.chunk_duration) + 1
        print(f"   Expected Chunks: {total_chunks}")
        
        self.is_running = True
        start_time = time.time()
        
        try:
            chunk_number = 0
            current_time = 0.0
            
            while current_time < audio_info['duration'] and self.is_running:
                print(f"\n⏰ Time: {current_time:.1f}s - Extracting chunk {chunk_number}")
                
                # Extract chunk
                chunk_info = self.audio_processor.extract_chunk(current_time, self.chunk_duration)
                
                if chunk_info:
                    # Process chunk (this will take some time due to AWS API)
                    result = await self.process_chunk(chunk_info, chunk_number)
                    
                    if result:
                        print(f"✅ Chunk {chunk_number} completed successfully")
                    else:
                        print(f"❌ Chunk {chunk_number} failed")
                else:
                    print(f"⚠️  No more audio data at {current_time:.1f}s")
                    break
                
                current_time += self.chunk_duration
                chunk_number += 1
                
                # Show progress
                progress = (current_time / audio_info['duration']) * 100
                print(f"📈 Progress: {progress:.1f}%")
            
            processing_time = time.time() - start_time
            print(f"\n🎉 TRANSCRIPTION COMPLETED!")
            print(f"   Total Processing Time: {processing_time:.1f} seconds")
            print(f"   Chunks Processed: {len(self.transcription_results)}")
            
            return self.transcription_results
            
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return self.transcription_results
        finally:
            # Cleanup S3 bucket
            self.transcribe_client.cleanup_s3_bucket()
    
    def stop_transcription(self):
        """Stop the transcription process"""
        self.is_running = False
    
    def get_full_transcript(self):
        """Get complete transcript from all chunks"""
        return ' '.join([r['transcript'] for r in self.transcription_results])
    
    def get_session_summary(self):
        """Get detailed session summary"""
        if not self.transcription_results:
            return "No transcription results"
        
        total_duration = sum([r['duration'] for r in self.transcription_results])
        avg_confidence = sum([r['confidence'] for r in self.transcription_results]) / len(self.transcription_results)
        
        return {
            'total_chunks': len(self.transcription_results),
            'total_duration': total_duration,
            'average_confidence': avg_confidence,
            'full_transcript': self.get_full_transcript(),
            'chunks': self.transcription_results
        }

# Demo callbacks
async def on_transcript_ready(result):
    """Callback when transcript is ready - integrate with LLM here"""
    print(f"📝 CALLBACK: New transcript ready for LLM analysis")
    print(f"   Chunk: {result['chunk_number']}")
    print(f"   Time: {result['start_time']:.1f}s")
    print(f"   Text: {result['transcript'][:100]}...")
    
    # HERE IS WHERE YOU'D INTEGRATE WITH YOUR LLM:
    # - Send transcript to LLM for analysis
    # - Generate real-time suggestions
    # - Update customer care agent UI
    # - Store in database

def on_chunk_completed(result):
    """Sync callback for chunk completion"""
    print(f"🔔 NOTIFICATION: Chunk {result['chunk_number']} processed")

# Main demo function
async def run_real_transcription_demo(audio_file_path):
    """Run the real AWS Transcribe demo"""
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file '{audio_file_path}' not found!")
        return
    
    # Create transcription manager
    manager = LiveTranscriptionManager(
        audio_file_path=audio_file_path,
        aws_region='us-east-1',  # Change to your region
        language_code='en-US',
        chunk_duration=30.0  # 30 seconds
    )
    
    # Add callbacks
    manager.add_callback(on_transcript_ready)
    manager.add_callback(on_chunk_completed)
    
    try:
        # Start transcription
        results = await manager.start_live_transcription()
        
        # Print summary
        summary = manager.get_session_summary()
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total Chunks: {summary['total_chunks']}")
        print(f"Average Confidence: {summary['average_confidence']:.2f}")
        print(f"Complete Transcript:")
        print(f"{'-'*40}")
        print(summary['full_transcript'])
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None

# Usage
if __name__ == "__main__":
    # Replace with your audio file
    audio_file = "call.wav"
    asyncio.run(run_real_transcription_demo(audio_file))

    # print("🚀 Real AWS Transcribe Demo Ready!")
    # print("Make sure you have:")
    # print("1. AWS credentials configured")
    # print("2. S3 and Transcribe permissions")
    # print("3. A WAV audio file")
    # print()
    # print("To run: asyncio.run(run_real_transcription_demo('your_file.wav'))")