import os
import io
import base64
import tempfile
from dotenv import load_dotenv
from PIL import Image
import gradio as gr
from groq import Groq

# --- NEW: PDF + RAG imports ---
import PyPDF2
import re
import math

# try semantic embeddings; fall back to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    _use_semantic = True
except Exception:
    _emb_model = None
    _use_semantic = False
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        TfidfVectorizer = None
        cosine_similarity = None

# SAFE TEMP DIRECTORY
gradio_temp = tempfile.mkdtemp(prefix="gradio_temp_")
os.environ["GRADIO_TEMP_DIR"] = gradio_temp
os.makedirs(gradio_temp, exist_ok=True)

# LOAD ENVIRONMENT VARIABLES
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")  # optional

# GROQ CLIENT
client = Groq(api_key=GROQ_API_KEY)

# SYSTEM PROMPTS
# (Image mode uses the original concise tone; PDF mode uses long-form)
system_prompt_image = (
    "You are a professional university coordinator"
    "Do not mention being an AI. Do not say 'In the image I see'; instead say "
    "'This university has'. Keep it to a maximum of ten sentences, "
    "no numbers or special characters. Start directly with the answer. "
    "Tell more about the university based on the image provided, such as which courses are offered, "
    "campus facilities, student life, any unique features that make this university stand out, "
    "and the credit system of the subjects."
)

system_prompt_pdf = (
    "You are a professional university coordinator answering questions using ONLY the provided document excerpts. "
    "Be accurate, specific, and helpful. If the document does not contain the answer, say so briefly. "
    "Cite sections by quoting short phrases when useful. Do not invent facts."
)

# IMAGE ENCODING FUNCTION (compressed to avoid large payloads)
def encode_image(image_path, max_size=(800, 800)):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=60)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# GROQ WHISPER TRANSCRIPTION
def transcribe_with_groq(audio_filepath):
    if not audio_filepath:
        return ""
    with open(audio_filepath, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            language="en"
        )
    return result.text

# ---------- PDF + RAG utilities ----------

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    # normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def chunk_text(text, max_tokens=450):
    # crude token proxy: ~4 chars per token
    max_chars = max_tokens * 4
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur.strip())
            # split long paragraph
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i+max_chars].strip())
            cur = ""
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if c]

def embed_chunks_semantic(chunks):
    embs = _emb_model.encode(chunks, normalize_embeddings=True)
    return np.array(embs)

def retrieve_top_k(chunks, question, k=5):
    if _use_semantic and _emb_model is not None:
        import numpy as np
        chunk_embs = embed_chunks_semantic(chunks)
        q_emb = _emb_model.encode([question], normalize_embeddings=True)
        sims = (chunk_embs @ q_emb[0])
        idx = np.argsort(-sims)[:k]
        return [chunks[i] for i in idx]
    else:
        # TF-IDF fallback
        if TfidfVectorizer is None:
            # no retrieval available; just return beginning
            return chunks[:k]
        docs = chunks + [question]
        vec = TfidfVectorizer(stop_words="english").fit(docs)
        M = vec.transform(docs)
        sims = cosine_similarity(M[-1], M[:-1]).ravel()
        order = sims.argsort()[::-1][:k]
        return [chunks[i] for i in order]

def build_context_from_pdf(pdf_path, question, k=5):
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return "", []
    chunks = chunk_text(text)
    top = retrieve_top_k(chunks, question, k=k)
    context = "\n\n---\n\n".join(top)
    return context, top

# IMAGE + QUERY ANALYSIS (unchanged)
def analyze_image_with_query(query, encoded_image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
    ]
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages
    )
    return response.choices[0].message.content

# TEXT TO SPEECH
def text_to_speech(text):
    output_path = os.path.join(gradio_temp, "final_output.mp3")
    try:
        from elevenlabs import ElevenLabs
        tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)
        audio = tts_client.generate(
            text=text,
            voice="Rachel",
            model="eleven_turbo_v2"
        )
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        return output_path
    except Exception:
        from gtts import gTTS
        tts = gTTS(text)
        tts.save(output_path)
        return output_path

# MAIN PROCESSING FUNCTION (updated for PDF RAG)
def process_inputs(audio_filepath, image_filepath, pdf_filepath):
    # Step 1: Speech to text
    stt_output = transcribe_with_groq(audio_filepath).strip()

    # Decide mode based on which file is provided
    coordinator_reply = ""

    # --- PDF mode (RAG) ---
    if pdf_filepath:
        # build retrieval context
        context, _ = build_context_from_pdf(pdf_filepath, stt_output, k=6)
        if context:
            user_msg = (
                f"Question: {stt_output}\n\n"
                f"Use the following document excerpts to answer. If uncertain, say you cannot find it.\n\n"
                f"{context}"
            )
        else:
            user_msg = f"Question: {stt_output}\n\nNo text could be extracted from the PDF."

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt_pdf},
                {"role": "user", "content": user_msg}
            ]
        )
        coordinator_reply = response.choices[0].message.content

    # --- Image mode (original behavior) ---
    elif image_filepath:
        encoded_img = encode_image(image_filepath)
        try:
            coordinator_reply = analyze_image_with_query(
                query=f"{system_prompt_image} The user asked: {stt_output}",
                encoded_image=encoded_img
            )
        except Exception as e:
            coordinator_reply = f"Sorry, I couldn't analyze the image due to an error: {e}"

    else:
        coordinator_reply = "Please upload an image or a PDF document so I can help."

    # Step 3: Text to speech
    voice_path = text_to_speech(coordinator_reply)

    return stt_output, coordinator_reply, voice_path

# GRADIO INTERFACE (adds PDF upload)
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(type="filepath", label="🎤 Speak to your coordinator"),
        gr.Image(type="filepath", label="📸 Upload campus or document image"),
        gr.File(type="filepath", label="📄 Upload a PDF document", file_types=[".pdf"])
    ],
    outputs=[
        gr.Textbox(label="🗣️ Speech to Text"),
        gr.Textbox(label="💬 Coordinator Response"),
        gr.Audio(label="🔊 Coordinator Voice")
    ],
    title="🎓 UNIQUERY: University Coordinator with Vision, Voice & Documents",
    allow_flagging="never"
)

iface.launch(share=True, debug=True, inbrowser=True)
