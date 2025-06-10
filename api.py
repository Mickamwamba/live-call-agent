from flask import Flask, request, jsonify
import tempfile
import os
import asyncio
from your_transcription_module import run_real_transcription_demo  # Adjust import path

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if not file.filename.endswith('.wav'):
        return jsonify({'error': 'Only .wav files are supported'}), 400

    try:
        # Save file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Run the async transcription process
        results = asyncio.run(run_real_transcription_demo(temp_audio_path))

        # Remove the temporary file
        os.unlink(temp_audio_path)

        if not results:
            return jsonify({'error': 'Transcription failed or returned no results'}), 500

        # Extract final transcript
        full_transcript = ' '.join([chunk['transcript'] for chunk in results])
        return jsonify({
            'transcript': full_transcript,
            'chunks': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
