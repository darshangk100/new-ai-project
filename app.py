from flask import Flask, request, jsonify
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from supabase import create_client
import google.generativeai as genai
from flask import render_template
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import uuid


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Stores memory per user sessiongit add .

user_sessions = {}

app.secret_key = '123456'  # Make sure this is strong in production
app.config['SESSION_TYPE'] = 'filesystem'  # Can use Redis later for scaling
Session(app)


# Setup API keysdjango-admin startproject myproject
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Chunking config
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Get text from PDF
def extract_text(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text

# Create embedding
def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response['embedding']

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    text = extract_text(file_path)
    chunks = splitter.split_text(text)
    print(f"Extracted {len(chunks)} chunks from the PDF.")
    # print(f"chunks>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {chunks}")

    for chunk in chunks:
        print(f"{chunk}")
        embedding = get_embedding(chunk)
        supabase.table("documents").insert({
            "chunk": chunk,
            "embedding": embedding
        }).execute()

    return jsonify({"status": "File uploaded and data embedded "})


@app.route('/ask', methods=['POST'])
def ask_question():
    user_id = "user_123"  # 👈 static user ID for all users

    # Create memory slot for this user if not present
    if user_id not in user_sessions:
        user_sessions[user_id] = []

    user_memory = user_sessions[user_id]
    user_question = request.json.get("question")

    # Step 1: Get embedding
    question_embedding = genai.embed_content(
        model="models/embedding-001",
        content=user_question,
        task_type="retrieval_query"
    )["embedding"]

    embedding_str = "[" + ",".join(map(str, question_embedding)) + "]"

    # Step 2: Retrieve matching chunks from Supabase
    response = supabase.rpc("match_chunks", {
        "query_embedding": embedding_str,
        "match_count": 3
    }).execute()

    if not response.data:
        return jsonify({"answer": "I couldn’t find that information in the document."}), 404

    context = "\n\n".join([item["chunk"] for item in response.data])

    # Step 3: Build prompt using context and history
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    chat = model.start_chat(history=user_memory)

    prompt = f"""
You are an intelligent and friendly assistant 🤖.

### Instructions:

- If the user's message is a greeting like "hi", "hello", or "hey", respond with a warm welcome like: "Hey there! 😊 How can I assist you today?"
- If the user includes a name (e.g., "Hi, I'm Darshan" or "My name is Darshan"), reply with: "Hello Darshan, how can I help you today? 😊"
- If the user says something like "no", "nothing", "not now", "thank you", or "bye", respond politely with a friendly closing such as: "You're welcome! Let me know if you need anything. 😊" or "Alright, have a great day! 👋"
- Otherwise, use the information from the context below to answer the user's question.
- Avoid using markdown syntax like **bold** or bullet symbols. Instead, use plain, clean formatting.

If the context does not contain enough information, reply with:
"I couldn't find that information."

### Context:
{context}

### Question:
{user_question}

### Answer:
"""

    gemini_response = chat.send_message(prompt)
    final_answer = gemini_response.text

    # Step 4: Add new messages to memory
    user_memory.append({"role": "user", "parts": [user_question]})
    user_memory.append({"role": "model", "parts": [final_answer]})

    # Step 5: Limit memory to last 5 user-bot pairs (10 messages)
    if len(user_memory) > 10:
        user_sessions[user_id] = user_memory[-10:]

    return jsonify({"answer": final_answer})


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':

    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)





