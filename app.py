from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import ask_rag

app = Flask(__name__)
CORS(app)

# -----------------------------
# ROUTE
# -----------------------------

@app.route("/")
def home():
    return "Backend running"


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json

    video_id = data.get("video_id")
    question = data.get("question")

    if not video_id or not question:
        return jsonify({"error": "Missing data"}), 400

    try:
        answer = ask_rag(video_id, question)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render dynamic port deta hai
    app.run(host="0.0.0.0", port=port)






"""imp things

{
  "video_id": "dQw4w9WgXcQ", == song
  "question": "summarize"
}

8ext9G7xspg ==  JavaScript Crash Course
kqtD5dpn9C8 == Python Full Course


versel link or live link = youtube-ai-assistant-rag.vercel.app

"""