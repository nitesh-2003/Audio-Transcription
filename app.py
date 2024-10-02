import os
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import threading
import time
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the MarianMT model and tokenizer for multi-language to English translation
model_name = 'Helsinki-NLP/opus-mt-mul-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Global variables
translated_text_global = ""
is_listening = False
recognizer = sr.Recognizer()

# Function to detect language
def detect_language(text):
    return detect(text)

# Function to translate text to English
def translate_to_english(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_text[0]

# Speech recognition and translation in a separate thread
def recognize_and_translate():
    global translated_text_global, is_listening

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)

        while is_listening:
            try:
                audio = recognizer.listen(source, timeout=5)
                speech_text = recognizer.recognize_google(audio)
                detected_language = detect_language(speech_text)

                if detected_language != 'en':
                    translated_text = translate_to_english(speech_text)
                else:
                    translated_text = speech_text

                translated_text_global = translated_text
                print(f"Translated: {translated_text}")

            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results: {e}")
            time.sleep(1)  # Add a delay to prevent rapid firing

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# API route to start listening
@app.route('/start_listening', methods=['POST'])
def start_listening():
    global is_listening
    is_listening = True
    threading.Thread(target=recognize_and_translate, daemon=True).start()
    return jsonify({'success': True})

# API route to stop listening
@app.route('/stop_listening', methods=['POST'])
def stop_listening():
    global is_listening
    is_listening = False
    return jsonify({'success': True})

# API route to fetch the latest translated text
@app.route('/get_translation', methods=['GET'])
def get_translation():
    global translated_text_global
    return jsonify({'translated_text': translated_text_global})

# API route to clear the translation text
@app.route('/clear_translation', methods=['POST'])
def clear_translation():
    global translated_text_global
    translated_text_global = ""  # Clear the global variable
    return jsonify({'success': True})

# API route to save transcription
@app.route('/save_transcription', methods=['POST'])
def save_transcription():
    global translated_text_global
    if not translated_text_global:
        return jsonify({'error': 'No transcription available to save.'}), 400

    # Ensure the 'transcriptions' folder exists
    transcriptions_folder = 'transcriptions'
    if not os.path.exists(transcriptions_folder):
        os.makedirs(transcriptions_folder)

    # Save the transcription to a file with a date and time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"transcription_{current_time}.txt"
    file_path = os.path.join(transcriptions_folder, file_name)
    
    try:
        with open(file_path, 'w') as file:
            file.write(translated_text_global)
        return jsonify({'success': True, 'file_path': file_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
