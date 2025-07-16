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
import tempfile
from docx import Document
from datetime import datetime
import re
import json

#this are all the liberies in the python and required to this project


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Stores memory per user sessiongit add .

user_sessions = {}

app.secret_key = '123456'#os.urandom(24)   # Make sure this is strong in production
app.config['SESSION_TYPE'] = 'filesystem'  # Can use Redis later for scaling
Session(app)


# Setup API keysdjango-admin startproject myproject
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Chunking config
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Get text from PDF
def extract_text_from_file(file_path, ext):
    if ext == ".pdf":
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
        return full_text
    # get the text from docx files
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    #get the text from txt files
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

embedding_model="models/embedding-001"

# Create embedding
def get_embedding(text):
    response = genai.embed_content(
        model=embedding_model,
        content=text,
        task_type="retrieval_document"
    )
    return response['embedding']

# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part in the request"}), 400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     #  Check if uploaded file is a PDF
#     if not file.filename.lower().endswith('.pdf'):
#         return jsonify({"error": " Invalid file type. Please upload a PDF file only."}), 400

#     # Optional: You can also check MIME type (extra safety)
#     if file.mimetype != 'application/pdf':
#         return jsonify({"error": " File type not supported. Only PDFs are allowed."}), 400

#     # Save file
#     file_path = os.path.join('uploads', file.filename)
#     file.save(file_path)

#     # Extract text and process
#     text = extract_text(file_path)
#     chunks = splitter.split_text(text)
#     print(f"Extracted {len(chunks)} chunks from the PDF.")

#     for chunk in chunks:
#         print(f"{chunk}")
#         embedding = get_embedding(chunk)
#         supabase.table("documents").insert({
#             "chunk": chunk,
#             "embedding": embedding
#         }).execute()

#     return jsonify({"status": " File uploaded and data embedded successfully!"})


@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    allowed_extensions = {'.pdf', '.docx', '.txt'}
    max_size_bytes = 50 * 1024 * 1024  # 50 MB

    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        return jsonify({"error": "Only PDF, DOCX, and TXT files are allowed."}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > max_size_bytes:
        return jsonify({"error": "File too large. Maximum allowed size is 50 MB."}), 400
    final_file_size = file_size  # Store original file size for metadata
    final_file_size_mb= file_size / (1024 * 1024)  # Convert to MB

    try:
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        # Step 1: Create metadata

        # metadata = {
        #     "filename": file.filename,
        #     "filesize": file_size,
        #     "user_id": session.get("user_id", "anonymous_user")
        # }

        # # Step 2: Save metadata to Supabase
        # # metadata_response = supabase.table("documents_metadata").insert(metadata).execute()

        # print(f"metadata{metadata_response}") 

        # if metadata_response is not None:
        #     print(f"metadata_response is storing in the super base")
        # else:
        #     return jsonify({"error": "Failed to store metadata in Supabase."}), 500


        # Step 3: Upload file to Supabase storage
        file_key = f"{file.filename}"
        with open(temp_path, "rb") as f:
            bucket = supabase.storage.from_("pdfs")

        # Delete if file already exists to avoid "resource already exists" error
            try:
                bucket.remove([file_key])
            except Exception:
                pass  # ignore if file doesn't exist

            # Upload file (no 'upsert' argument)
            storage_response = bucket.upload(
                path=file_key,
                file=f,
                file_options={"content-type": "application/octet-stream"}
            )


        if storage_response.status_code != 200 and storage_response.status_code != 201:    
            return jsonify({"error": f"Failed to upload to Supabase storage. Status code: {storage_response.status_code}"}), 500
        

        # Step 4: Extract text and insert embeddings
        raw_text = extract_text_from_file(temp_path, ext)

        cleaned_text = re.sub(r"[^\w\s,.!?;:()-]", "", raw_text)

        chunks = splitter.split_text(cleaned_text)

        total_chunks = len(chunks)

        print(f"Extracted {total_chunks} chunks from the document.")

        model = genai.GenerativeModel("models/gemini-1.5-flash")
        chat = model.start_chat()

        classification_prompt = f"""
        Classify the document below. Categories:
        - Industry/Sector
        - Geographic Region
        - Service Type / Proposal Category
        - Document Type (e.g., RFP, Technical Proposal, NDA, MOU, Contract, Resume, Product Overview)

        Return JSON:
        {{
        "industry": [],
        "region": [],
        "service_type": [],
        "document_type": ""
        }}

        Document:
        \"\"\"{cleaned_text}\"\"\"
        """
        classification_response = chat.send_message(classification_prompt)
        raw_text = classification_response.text.strip()

        print(f"Classification response: {raw_text}")

            # Remove markdown code block wrappers ```json ... ```
        cleaned_text = re.sub(r"^```(?:json)?\s*|```$", "", raw_text, flags=re.DOTALL).strip()

        print(f"Cleaned classification response: {cleaned_text}")

        classification_json = json.loads(cleaned_text)

        industry= classification_json.get("industry", [])

        region= classification_json.get("region", [])

        service_type= classification_json.get("service_type", [])


        # details=supabase.table("documents").insert({
        #         "industry": industry,
        #         "region": region,
        #         "service_type": service_type,
        #     })
        print(f"Classification JSON: {classification_json}")

        document_type= classification_json.get("document_type", "")

        document_info = supabase.table("documents ").insert({
                    "file_name": file.filename,
                    "file_type": ext.replace(".", ""),  # pdf, docx, txt
                    "file_size": final_file_size,
                    "file_size_mb": final_file_size_mb,
                    "total_chunks":total_chunks,
                    "document_type": document_type,
                }).execute()

        for index, chunk in enumerate(chunks): 

            embedding = get_embedding(chunk)
            
            embedded_table= supabase.table("document_embedding").insert({"embedding":embedding,"model":embedding_model}).execute()

            response = supabase.table("document_chunks").insert({
            "chunk_index": index,      
            "chunks": chunk
        }).execute()
        print(f"Chunk {index} inserted into documents: {response}")
                

        # Step 5: AI Classification (No summarization)
        try:

                # Save classification to Supabase
            supabase.table("documents").insert(classification_json).execute()

        except Exception as e:
            print("Classification failed:", e)
            print(f"Fallback classification JSON: {classification_json}")
        
        return jsonify({
            "status": "File uploaded, metadata stored, embeddings and classification done!",
            "classification": classification_json
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500







@app.route('/ask', methods=['POST'])
def ask_question():
    user_id = "user_123"  # Static user ID for all users

    # if 'user_id' not in session:
    #     session['user_id'] = str(uuid.uuid4())
    #     user_id = session['user_id']                # dynami user id we can replace witht the  user_id = "user_123" 


    # Step 0: Validate incoming JSON and question field
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    user_question = data["question"].strip()
    if not user_question:
        return jsonify({"error": "Question cannot be empty"}), 400

    # Step 1: Get embedding from Gemini
    try:
        question_embedding = genai.embed_content(
            model="models/embedding-001",
            content=user_question,
            task_type="retrieval_query"
        )["embedding"]
    except Exception as e:
        return jsonify({"error": f"Failed to generate embedding: {str(e)}"}), 500

    embedding_str = "[" + ",".join(map(str, question_embedding)) + "]"

    # Step 2: Retrieve matching chunks from Supabase
    try:
        response = supabase.rpc("match_chunks", {
            "query_embedding": embedding_str,
            "match_count": 3
        }).execute()
    except Exception as e:
        return jsonify({"error": f"Supabase query failed: {str(e)}"}), 500

    if not response.data:
        return jsonify({"answer": "I couldn't find that information in the document."}), 404

    context_chunks = [item.get("chunk", "") for item in response.data if "chunk" in item]
    context = "\n\n".join(context_chunks).strip()

    if not context:
        return jsonify({"answer": "No usable context found from the database."}), 404

    # Step 3: Build prompt using context and chat history
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        # Create memory if not present
        if user_id not in user_sessions:
            user_sessions[user_id] = []
        user_memory = user_sessions[user_id]

        chat = model.start_chat(history=user_memory)

        prompt = f"""
You are an intelligent and friendly assistant.

### Instructions:
- If the user's message is a greeting like "hi", "hello", or "hey", respond with a warm welcome like: "Hey there! How can I assist you today?"
- If the user includes a name (e.g., "Hi, I'm Darshan" or "My name is Darshan"), reply with: "Hello Darshan, how can I help you today?"
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
        final_answer = gemini_response.text.strip()

    except Exception as e:
        return jsonify({"error": f"Gemini model failed to respond: {str(e)}"}), 500

    # Step 4: Save memory
    user_memory.append({"role": "user", "parts": [user_question]})
    user_memory.append({"role": "model", "parts": [final_answer]})

    # Step 5: Limit memory to last 5 user-bot pairs (10 messages)
    if len(user_memory) > 10:
        user_sessions[user_id] = user_memory[-10:]

    return jsonify({"answer": final_answer})


# if we add the dynami user id we need to use this ask api ---------------ask api with dynamic user id

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     #  Generate or get session-based user ID
#     if 'user_id' not in session:
#         session['user_id'] = str(uuid.uuid4())
#     user_id = session['user_id']

#     data = request.get_json()
#     if not data or "question" not in data:
#         return jsonify({"error": "Missing 'question' in request body"}), 400

#     user_question = data["question"].strip()
#     if not user_question:
#         return jsonify({"error": "Question cannot be empty"}), 400

#     try:
#         question_embedding = genai.embed_content(
#             model="models/embedding-001",
#             content=user_question,
#             task_type="retrieval_query"
#         )["embedding"]
#     except Exception as e:
#         return jsonify({"error": f"Failed to generate embedding: {str(e)}"}), 500

#     embedding_str = "[" + ",".join(map(str, question_embedding)) + "]"

#     try:
#         response = supabase.rpc("match_chunks", {
#             "query_embedding": embedding_str,
#             "match_count": 3
#         }).execute()
#     except Exception as e:
#         return jsonify({"error": f"Supabase query failed: {str(e)}"}), 500

#     if not response.data:
#         return jsonify({"answer": "I couldn't find that information in the document."}), 404

#     context_chunks = [item.get("chunk", "") for item in response.data if "chunk" in item]
#     context = "\n\n".join(context_chunks).strip()

#     if not context:
#         return jsonify({"answer": "No usable context found from the database."}), 404

#     try:
#         model = genai.GenerativeModel("models/gemini-1.5-flash")

#         #  Session-based memory management
#         if user_id not in user_sessions:
#             user_sessions[user_id] = []
#         user_memory = user_sessions[user_id]

#         chat = model.start_chat(history=user_memory)

#         prompt = f"""
# You are an intelligent and friendly assistant.

# ### Instructions:
# - If the user's message is a greeting like "hi", "hello", or "hey", respond with a warm welcome like: "Hey there! How can I assist you today?"
# - If the user includes a name (e.g., "Hi, I'm Darshan" or "My name is Darshan"), reply with: "Hello Darshan, how can I help you today?"
# - If the user says something like "no", "nothing", "not now", "thank you", or "bye", respond politely with a friendly closing such as: "You're welcome! Let me know if you need anything. 😊" or "Alright, have a great day! 👋"
# - Otherwise, use the information from the context below to answer the user's question.
# - Avoid using markdown syntax like **bold** or bullet symbols. Instead, use plain, clean formatting.

# If the context does not contain enough information, reply with:
# "I couldn't find that information."

# ### Context:
# {context}

# ### Question:
# {user_question}

# ### Answer:
# """
#         gemini_response = chat.send_message(prompt)
#         final_answer = gemini_response.text.strip()

#     except Exception as e:
#         return jsonify({"error": f"Gemini model failed to respond: {str(e)}"}), 500

#     user_memory.append({"role": "user", "parts": [user_question]})
#     user_memory.append({"role": "model", "parts": [final_answer]})

#     if len(user_memory) > 10:
#         user_sessions[user_id] = user_memory[-10:]

#     return jsonify({"answer": final_answer})




@app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':

    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)





