import wave
import tempfile
import os

class RealTimeAudioProcessor:
    def __init__(self, audio_file_path, chunk_duration=30.0):
        self.audio_file_path = audio_file_path
        self.chunk_duration = chunk_duration

    def get_audio_info(self):
        with wave.open(self.audio_file_path, 'rb') as wav:
            return {
                'sample_rate': wav.getframerate(),
                'channels': wav.getnchannels(),
                'sample_width': wav.getsampwidth(),
                'total_frames': wav.getnframes(),
                'duration': wav.getnframes() / wav.getframerate()
            }

    def extract_chunk(self, start_time):
        with wave.open(self.audio_file_path, 'rb') as wav:
            rate = wav.getframerate()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            start = int(start_time * rate)
            frames = int(self.chunk_duration * rate)
            wav.setpos(start)
            data = wav.readframes(frames)

            if not data:
                return None

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with wave.open(tmp.name, 'wb') as out:
                out.setnchannels(channels)
                out.setsampwidth(width)
                out.setframerate(rate)
                out.writeframes(data)

            return {
                'file_path': tmp.name,
                'start_time': start_time,
                'duration': len(data) / (rate * channels * width)
            }
