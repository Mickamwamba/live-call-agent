import boto3
import time
import requests
from botocore.exceptions import ClientError


class AudioTranscriber:
    def __init__(self, region_name='us-east-1'):
        """
        Initialize AWS clients for S3 and Transcribe services.
        
        Args:
            region_name (str): AWS region name (e.g., 'us-east-1', 'us-west-2')
                              Must match the region where your S3 bucket exists
        """
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.transcribe_client = boto3.client('transcribe', region_name=region_name)
    
    def upload_file(self, file_path, bucket_name, s3_key=None):
        """
        Upload a file to S3 bucket.
        
        Args:
            file_path (str): Local path to the file
            bucket_name (str): Name of the S3 bucket
            s3_key (str): S3 object key (defaults to filename if None)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if s3_key is None:
            s3_key = file_path.split('/')[-1]
        
        try:
            self.s3_client.upload_file(file_path, bucket_name, s3_key)
            print(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return False
    
    def job_exists(self, job_name):
        """
        Check if a transcription job exists.
        
        Args:
            job_name (str): Name of the transcription job
        
        Returns:
            dict: Job details if exists, None if doesn't exist
        """
        try:
            response = self.transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            return response['TranscriptionJob']
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['BadRequestException', 'NotFoundException']:
                # Job doesn't exist
                return None
            else:
                print(f"Error checking job existence: {e}")
                return None
    
    def start_transcription(self, job_name, bucket_name, file_name, language_code='en-US'):
        """
        Start a transcription job (only if it doesn't already exist).
        
        Args:
            job_name (str): Unique name for the transcription job
            bucket_name (str): S3 bucket name
            file_name (str): S3 object key
            language_code (str): Language code for transcription
        
        Returns:
            str: Job status ('STARTED', 'EXISTS', 'ERROR')
        """
        # Check if job already exists
        existing_job = self.job_exists(job_name)
        if existing_job:
            status = existing_job['TranscriptionJobStatus']
            print(f"Transcription job '{job_name}' already exists with status: {status}")
            return 'EXISTS'
        
        job_uri = f"s3://{bucket_name}/{file_name}"
        
        # Determine media format from file extension
        file_extension = file_name.split('.')[-1].lower()
        supported_formats = ['mp3', 'mp4', 'wav', 'flac', 'm4a', 'ogg', 'amr', 'webm']
        media_format = file_extension if file_extension in supported_formats else 'mp3'
        
        try:
            self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': job_uri},
                MediaFormat=media_format,
                LanguageCode=language_code
            )
            print(f"Started transcription job: {job_name}")
            return 'STARTED'
        except ClientError as e:
            print(f"Error starting transcription job: {e}")
            return 'ERROR'
    
    def wait_for_completion(self, job_name, polling_interval=10, max_wait_time=3600):
        """
        Wait for transcription job to complete.
        
        Args:
            job_name (str): Name of the transcription job
            polling_interval (int): Seconds to wait between status checks
            max_wait_time (int): Maximum time to wait in seconds (default: 1 hour)
        
        Returns:
            str: Job status ('COMPLETED', 'FAILED', 'TIMEOUT', or 'ERROR')
        """
        print("Waiting for transcription to complete...")
        start_time = time.time()
        
        while True:
            # Check if we've exceeded max wait time
            if time.time() - start_time > max_wait_time:
                print(f"Timeout: Job did not complete within {max_wait_time} seconds")
                return 'TIMEOUT'
            
            try:
                response = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                status = response['TranscriptionJob']['TranscriptionJobStatus']
                
                if status == 'COMPLETED':
                    print("Transcription completed successfully!")
                    return status
                elif status == 'FAILED':
                    failure_reason = response['TranscriptionJob'].get('FailureReason', 'Unknown error')
                    print(f"Transcription failed: {failure_reason}")
                    return status
                elif status == 'IN_PROGRESS':
                    elapsed_time = int(time.time() - start_time)
                    print(f"Transcription in progress... (elapsed: {elapsed_time}s)")
                    time.sleep(polling_interval)
                else:
                    print(f"Unknown status: {status}")
                    time.sleep(polling_interval)
                    
            except ClientError as e:
                print(f"Error checking job status: {e}")
                return 'ERROR'
    
    def get_transcript(self, job_name):
        """
        Get the transcript text from a completed transcription job.
        
        Args:
            job_name (str): Name of the transcription job
        
        Returns:
            str: Transcript text or None if error
        """
        try:
            response = self.transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            
            job_status = response['TranscriptionJob']['TranscriptionJobStatus']
            
            # Check if job is completed
            if job_status != 'COMPLETED':
                print(f"Transcription job is not completed. Current status: {job_status}")
                return None
            
            # Get transcript URL
            transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
            
            # Download and parse transcript
            transcript_response = requests.get(transcript_uri, timeout=30)
            transcript_response.raise_for_status()
            
            transcript_json = transcript_response.json()
            
            # Extract transcript text
            if 'results' in transcript_json and 'transcripts' in transcript_json['results']:
                transcripts = transcript_json['results']['transcripts']
                if transcripts and len(transcripts) > 0:
                    transcript_text = transcripts[0]['transcript']
                    return transcript_text
                else:
                    print("No transcripts found in the results")
                    return None
            else:
                print("Invalid transcript format")
                return None
            
        except ClientError as e:
            print(f"Error getting transcript: {e}")
            return None
        except requests.RequestException as e:
            print(f"Error downloading transcript: {e}")
            return None
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error parsing transcript JSON: {e}")
            return None
    
    def get_or_create_transcript(self, job_name, bucket_name=None, file_name=None, language_code='en-US'):
        """
        Get transcript from existing job or create new job if it doesn't exist.
        
        Args:
            job_name (str): Name of the transcription job
            bucket_name (str): S3 bucket name (required if job doesn't exist)
            file_name (str): S3 object key (required if job doesn't exist)
            language_code (str): Language code for transcription
        
        Returns:
            str: Transcript text or None if error
        """
        # Check if job already exists
        existing_job = self.job_exists(job_name)
        
        if existing_job:
            status = existing_job['TranscriptionJobStatus']
            print(f"Found existing job '{job_name}' with status: {status}")
            
            if status == 'COMPLETED':
                return self.get_transcript(job_name)
            elif status == 'FAILED':
                failure_reason = existing_job.get('FailureReason', 'Unknown error')
                print(f"Existing job failed: {failure_reason}")
                return None
            elif status == 'IN_PROGRESS':
                # Wait for existing job to complete
                final_status = self.wait_for_completion(job_name)
                if final_status == 'COMPLETED':
                    return self.get_transcript(job_name)
                else:
                    return None
        else:
            # Job doesn't exist, create it
            if not bucket_name or not file_name:
                print("Job doesn't exist and bucket_name/file_name not provided to create new job")
                return None
            
            print(f"Job '{job_name}' doesn't exist, creating new job...")
            start_result = self.start_transcription(job_name, bucket_name, file_name, language_code)
            
            if start_result == 'STARTED':
                final_status = self.wait_for_completion(job_name)
                if final_status == 'COMPLETED':
                    return self.get_transcript(job_name)
            
            return None
    
    def delete_transcription_job(self, job_name):
        """
        Delete a transcription job (cleanup).
        
        Args:
            job_name (str): Name of the transcription job to delete
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.transcribe_client.delete_transcription_job(
                TranscriptionJobName=job_name
            )
            print(f"Successfully deleted transcription job: {job_name}")
            return True
        except ClientError as e:
            print(f"Error deleting transcription job: {e}")
            return False
    
    def list_transcription_jobs(self, status_filter=None, max_results=50):
        """
        List transcription jobs.
        
        Args:
            status_filter (str): Filter by status ('IN_PROGRESS', 'FAILED', 'COMPLETED')
            max_results (int): Maximum number of jobs to return
        
        Returns:
            list: List of transcription job summaries
        """
        try:
            params = {'MaxResults': max_results}
            if status_filter:
                params['Status'] = status_filter
            
            response = self.transcribe_client.list_transcription_jobs(**params)
            return response['TranscriptionJobSummaries']
        except ClientError as e:
            print(f"Error listing transcription jobs: {e}")
            return []
    
def main():
    """Main function"""
    
    # Configuration
    BUCKET_NAME = 'impetus-hackathon'
    FILE_PATH = 'call.mp3'
    JOB_NAME = 'test-transcription-jobX'
    REGION = 'us-east-1' 
    
    # Initialize transcriber
    transcriber = AudioTranscriber(region_name=REGION)
    
    transcript = transcriber.get_or_create_transcript(
        job_name=JOB_NAME,
        bucket_name=BUCKET_NAME,
        file_name=FILE_PATH.split('/')[-1]
    )
    if transcript:
        print("Transcribed Text:", transcript)
    else:
        print("Failed to get transcript")

if __name__ == "__main__":
    main()