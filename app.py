import boto3
import time
import os
import requests

def uploadFile(bucket_name,file_name):

    s3 = boto3.client('s3')
    s3.upload_file(file_name, bucket_name, file_name)
    return; 



def transcribe(job_name,bucket_name,file_name):
    transcribe = boto3.client('transcribe')
    
    job_uri = f"s3://{bucket_name}/{file_name}"

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='mp3',  # or mp3
        LanguageCode='en-US'
    )
    return transcribe



job_name = "test-transcription-job"
bucket_name = 'impetus-hackathon'
file_name = 'call.mp3'
client = boto3.client('transcribe')


# uploadFile(bucket_name,file_name)
# transribe = transcribe(job_name,bucket_name,file_name)

# while True:
#     status = transcribe.GetTranscriptionJob(TranscriptionJobName=job_name)
#     if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
#         break
#     print("Waiting for transcription...")
#     time.sleep(5)

# Get the transcript
response = client.get_transcription_job(TranscriptionJobName=job_name)

# Extract the Transcript URL
transcription_url = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
# print("Transcript URL:", transcription_url)

# Download and display transcript

transcript_json = requests.get(transcription_url).json()
text = transcript_json['results']['transcripts'][0]['transcript']
print("Transcribed Text:", text)

