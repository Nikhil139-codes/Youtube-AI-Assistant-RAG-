# -----------------------------
# IMPORTS
# -----------------------------
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# 🔑 GROQ API KEY
# -----------------------------


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# 🔥 CACHE (FAST RESPONSE)
# -----------------------------
vector_store_cache = {}

# -----------------------------
# 📺 LOAD VIDEO (Hindi + English)
# -----------------------------
def load_video(video_id):

    # Cache check
    if video_id in vector_store_cache:
        return vector_store_cache[video_id]

    try:
        api = YouTubeTranscriptApi()

        # 🔥 Step 1: Get all transcripts
        transcript_list = api.list(video_id)

        # 🔥 Step 2: Try Hindi first
        try:
            transcript_obj = transcript_list.find_transcript(['hi'])
            print("✅ Hindi transcript loaded")
        except:
            # 🔥 Step 3: fallback to English
            try:
                transcript_obj = transcript_list.find_transcript(['en'])
                print("✅ English transcript loaded")
            except:
                # 🔥 Step 4: fallback to ANY language
                transcript_obj = transcript_list.find_transcript(
                    [t.language_code for t in transcript_list]
                )
                print("⚠ Using other language transcript")

        # 🔥 Step 5: Fetch transcript
        fetched = transcript_obj.fetch()
        transcript = " ".join(chunk.text for chunk in fetched)

    except Exception as e:
        raise Exception("❌ No transcript available for this video")

    if not transcript.strip():
        raise Exception("❌ Empty transcript")

    # -----------------------------
    # ✂ Split text
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript])

    # -----------------------------
    # 🔎 Embeddings
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(docs, embeddings)

    # Cache store
    vector_store_cache[video_id] = vector_store

    return vector_store


# -----------------------------
# 🤖 ASK FUNCTION
# -----------------------------
def ask_rag(video_id, question):

    try:
        vector_store = load_video(video_id)

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        if not docs:
            return "❌ No relevant content found"

        context = "\n\n".join(doc.page_content for doc in docs)

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
        return str(e)


# -----------------------------
# 💬 TEST (OPTIONAL)
# -----------------------------
if __name__ == "__main__":

    while True:
        video_id = input("\n🎥 Enter video ID: ")
        question = input("💬 Ask: ")

        if question.lower() == "exit":
            break

        answer = ask_rag(video_id, question)
        print("\n🤖 Answer:", answer)