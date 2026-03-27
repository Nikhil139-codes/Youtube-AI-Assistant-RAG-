# -----------------------------
# ✅ IMPORTS
# -----------------------------
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

# -----------------------------
# 🔑 GROQ API KEY
# -----------------------------


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# 🎥 VIDEO ID
# -----------------------------
video_id = "dQw4w9WgXcQ"

# -----------------------------
# 📜 GET TRANSCRIPT
# -----------------------------
try:
    api = YouTubeTranscriptApi()
    transcript_data = api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_data)
    print("✅ Transcript fetched")
except Exception as e:
    print("❌ Error:", e)
    exit()

if not transcript.strip():
    print("❌ Empty transcript")
    exit()

# -----------------------------
# ✂️ SPLIT TEXT
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = splitter.create_documents([transcript])

# -----------------------------
# 🔍 EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 🧠 VECTOR STORE
# -----------------------------
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# -----------------------------
# 🤖 GROQ LLM (FINAL 🔥)
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # 🔥 best free model
    api_key=GROQ_API_KEY,
    temperature=0.3
)

# -----------------------------
# 📝 PROMPT
# -----------------------------
prompt = PromptTemplate(
    template="""
Context:
{context}

Question:
{question}

Give a short and clear answer in 2-3 lines only.
If not found, say: I don't know.
""",
    input_variables=["context", "question"]
)

# -----------------------------
# 🔄 FORMAT DOCS
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------
# 🔗 RAG CHAIN
# -----------------------------
chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
}) | prompt | llm | StrOutputParser()

# -----------------------------
# 💬 CHAT LOOP
# -----------------------------
while True:
    query = input("\n💬 Ask something (type 'exit'): ")

    if query.lower() == "exit":
        break

    answer = chain.invoke(query)
    print("\n🤖 Answer:\n", answer)