import os
import tempfile
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
from googletrans import Translator

app = Flask(__name__)

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert_audio_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        audio_file_path = os.path.join(temp_dir, audio_file.filename)
        audio_file.save(audio_file_path)

        # Convert the file to WAV format if necessary
        if not audio_file_path.endswith('.wav'):
            audio_file_path = convert_to_wav(audio_file_path)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            # Recognize speech with automatic language detection
            text = recognizer.recognize_google(audio_data, language='auto')

        # Translate the text to English
        translator = Translator()
        translated_text = translator.translate(text, dest='en').text

        return jsonify({'text': translated_text})
    except sr.UnknownValueError:
        return jsonify({'error': 'Speech recognition could not understand the audio'}), 400
    except sr.RequestError:
        return jsonify({'error': 'Could not request results from the speech recognition service'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

@app.route('/save', methods=['POST'])
def save_transcription():
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400

    text = request.form['text']
    with open('transcription.txt', 'w') as file:
        file.write(text)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)