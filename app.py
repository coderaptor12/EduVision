from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

load_dotenv()

from explain import explain_topic
from generate_images import generate_image
from voice_generator import generate_voice

app = Flask(__name__)

# Ensure required folders exist
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/audio", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/explain", methods=["POST"])
def topic():
    topic_name = request.form.get("topic", "").strip()
    level = request.form.get("level", "diploma").strip()
    voice = request.form.get("voice", "en-IN-NeerjaNeural").strip()

    if not topic_name:
        return "Topic is required", 400

    explanation = explain_topic(topic_name, level)

    safe_topic = topic_name.replace(" ", "_")
    image_path = f"static/images/{safe_topic}.png"
    audio_path = f"static/audio/{safe_topic}.mp3"

    success = generate_image(topic_name, explanation, image_path)

    if not success:
        return render_template(
            "topic.html",
            topic=topic_name,
            level=level,
            voice=voice,
            explanation=explanation,
            image=None,
            audio=None,
            error="Image generation failed. HuggingFace did not return an image."
        )

    # Generate voice in selected style
    generate_voice(explanation, audio_path, voice)

    return render_template(
        "topic.html",
        topic=topic_name,
        level=level,
        voice=voice,
        explanation=explanation,
        image=image_path,
        audio=audio_path
    )

if __name__ == "__main__":
    app.run(debug=True)
