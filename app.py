from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64
import pandas as pd
import plotly.express as px
import os

app = Flask(__name__)

# CSV file to store emotion data
CSV_FILE = 'emotion_log.csv'

# Ensure CSV exists with headers
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=['emotion'])
    df.to_csv(CSV_FILE, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json.get('image', None)
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    imgdata = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
    except Exception as e:
        emotion = "No Face Detected"

    # Save emotion to CSV
    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame({'emotion': [emotion]})], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    return jsonify({'emotion': emotion})

@app.route('/dashboard')
def dashboard():
    # Load CSV, fallback to default data if missing or empty
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame({'emotion': ['Happy', 'Sad', 'Neutral'], 'count': [1,1,1]})
    else:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            df = pd.DataFrame({'emotion': ['Happy', 'Sad', 'Neutral'], 'count': [1,1,1]})

    # Count emotions
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']

    # Create pie chart
    fig = px.pie(emotion_counts, names='emotion', values='count', title='Emotion Analysis Dashboard')
    graph_html = fig.to_html(full_html=False)

    return render_template('dashboard.html', graph_html=graph_html)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render uses this port
    app.run(host="0.0.0.0", port=port, debug=True)