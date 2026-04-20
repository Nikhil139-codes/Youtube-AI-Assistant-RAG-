from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def create_faiss_index(video_id):
    print(f"Fetching transcript for video: {video_id}...")
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Try Hindi first
        try:
            transcript_obj = transcript_list.find_transcript(['hi'])
            print("✅ Hindi transcript loaded")
        except:
            # Fallback to English
            try:
                transcript_obj = transcript_list.find_transcript(['en'])
                print("✅ English transcript loaded")
            except:
                # Fallback to ANY language
                transcript_obj = transcript_list.find_transcript(
                    [t.language_code for t in transcript_list]
                )
                print("⚠ Using other language transcript")
                
        fetched = transcript_obj.fetch()
        transcript = " ".join(chunk.text for chunk in fetched)
        
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return

    if not transcript.strip():
        print("❌ Transcript is empty.")
        return

    # Split text
    print("✂️ Splitting transcript into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.create_documents([transcript])
    
    # Generate embeddings and VectorStore
    print("🧠 Generating embeddings and creating FAISS index (This may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save locally
    folder_path = f"faiss_index/{video_id}"
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path)
    print(f"🎉 Success! FAISS index saved to {folder_path}!")

if __name__ == "__main__":
    print("🚀 Youtube RAG - Database Precomputation Tool")
    vid = input("🎥 Enter YouTube Video ID: ")
    create_faiss_index(vid)
