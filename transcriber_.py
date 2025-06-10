import boto3
import uuid
import time
import json
import urllib.request
from botocore.exceptions import ClientError

class AWSTranscribeClient:
    def __init__(self, region='us-east-1', language_code='en-US'):
        self.client = boto3.client('transcribe', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = f"transcribe-temp-{uuid.uuid4().hex[:8]}"
        self.language_code = language_code
        self.region = region
        self._create_bucket()

    def _create_bucket(self):
        try:
            config = {'LocationConstraint': self.region} if self.region != 'us-east-1' else {}
            self.s3.create_bucket(Bucket=self.bucket, **({'CreateBucketConfiguration': config} if config else {}))
        except ClientError as e:
            if e.response['Error']['Code'] != 'BucketAlreadyExists':
                raise

    def upload_audio(self, file_path, key):
        self.s3.upload_file(file_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def start_job(self, job_name, s3_uri):
        return self.client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='wav',
            LanguageCode=self.language_code,
            Settings={'ShowSpeakerLabels': True, 'MaxSpeakerLabels': 2}
        )['TranscriptionJob']

    def wait_for_completion(self, job_name, timeout=90):
        start = time.time()
        while time.time() - start < timeout:
            job = self.client.get_transcription_job(TranscriptionJobName=job_name)['TranscriptionJob']
            status = job['TranscriptionJobStatus']
            if status == 'COMPLETED':
                return job
            elif status == 'FAILED':
                raise RuntimeError(f"Job failed: {job.get('FailureReason')}")
            time.sleep(2)
        raise TimeoutError("Transcription timed out.")

    def get_result(self, job):
        uri = job['Transcript']['TranscriptFileUri']
        with urllib.request.urlopen(uri) as res:
            data = json.loads(res.read().decode())
        return {
            'transcript': data['results']['transcripts'][0]['transcript'],
            'segments': data['results'].get('speaker_labels', {}).get('segments', [])
        }

    def cleanup(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def delete_bucket(self):
        objs = self.s3.list_objects_v2(Bucket=self.bucket).get('Contents', [])
        for o in objs:
            self.s3.delete_object(Bucket=self.bucket, Key=o['Key'])
        self.s3.delete_bucket(Bucket=self.bucket)
