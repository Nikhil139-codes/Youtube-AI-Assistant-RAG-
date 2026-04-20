# -----------------------------
# IMPORTS
# -----------------------------
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# 🔑 GROQ API KEY
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# 🔥 GLOBAL EMBEDDINGS (Init once)
# -----------------------------
# Initialized globally to remove heavy creation during request time
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
except Exception as e:
    print("Warning: Could not load embeddings model:", e)
    embeddings = None

# -----------------------------
# 🔥 CACHE (FAST RESPONSE)
# -----------------------------
vector_store_cache = {}

# -----------------------------
# 📜 FALLBACK: FETCH TRANSCRIPT
# -----------------------------
def fetch_fallback_transcript(video_id):
    """Fetches the transcript and truncates to ~4000 chars for simple fallback mode."""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Try Hindi first
        try:
            transcript_obj = transcript_list.find_transcript(['hi'])
            print("✅ Hindi transcript loaded (Fallback)")
        except:
            # Fallback to English
            try:
                transcript_obj = transcript_list.find_transcript(['en'])
                print("✅ English transcript loaded (Fallback)")
            except:
                # Fallback to any available language
                transcript_obj = transcript_list.find_transcript(
                    [t.language_code for t in transcript_list]
                )
                print("⚠ Using other language transcript (Fallback)")
                
        fetched = transcript_obj.fetch()
        transcript = " ".join(chunk.text for chunk in fetched)
        
        # Truncate to first 4000 characters to avoid exceeding simple LLM context bounds
        return transcript[:4000]
        
    except Exception as e:
        print(f"Fallback transcript error: {e}")
        return None

# -----------------------------
# 📺 LOAD VIDEO (FAISS OR FALLBACK)
# -----------------------------
def load_video_index(video_id):
    """Loads precomputed FAISS index from local folder."""
    # Cache check
    if video_id in vector_store_cache:
        return vector_store_cache[video_id]

    folder_path = f"faiss_index/{video_id}"
    
    if os.path.exists(folder_path) and embeddings is not None:
        try:
            print(f"✅ Loading precomputed FAISS index for {video_id}")
            vector_store = FAISS.load_local(
                folder_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vector_store_cache[video_id] = vector_store
            return vector_store
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            return None
    else:
        print(f"⚠ FAISS index {folder_path} not found. Operating in fallback mode.")
        return None

# -----------------------------
# 🤖 ASK FUNCTION
# -----------------------------
def ask_rag(video_id, question):
    try:
        if not GROQ_API_KEY:
            return "❌ Missing GROQ_API_KEY."

        # Attempt to load precomputed RAG index
        vector_store = load_video_index(video_id)
        context = ""
        
        if vector_store:
            # Precomputed RAG mode
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
            
            if not docs:
                context = "No relevant context found in video."
            else:
                context = "\n\n".join(doc.page_content for doc in docs)
        else:
            # Fallback Simple Mode
            transcript = fetch_fallback_transcript(video_id)
            if transcript:
                context = transcript
            else:
                return "❌ No precomputed index found, and fallback transcript could not be fetched."

        # -----------------------------
        # 🤖 LLM (Groq)
        # -----------------------------
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0.3
        )

        prompt = f"""
You are an AI assistant.

Context:
{context}

Question:
{question}

Answer in same language as the question.
Keep answer short (2-3 lines).
If not found, say: I don't know.
"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        print(f"Error in ask_rag: {e}")
        return "❌ An internal error occurred while processing your request."

# -----------------------------
# 💬 TEST (OPTIONAL)
# -----------------------------
if __name__ == "__main__":
    print("Test mode. Make sure faiss_index/<video_id>/ is populated if testing RAG.")
    while True:
        vid_input = input("\n🎥 Enter video ID: ")
        q_input = input("💬 Ask: ")

        if q_input.lower() == "exit":
            break

        ans = ask_rag(vid_input, q_input)
        print("\n🤖 Answer:", ans)