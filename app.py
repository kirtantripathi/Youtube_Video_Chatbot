from fastapi import FastAPI, HTTPException
import re
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import List
from pytube import YouTube
import whisper
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import subprocess
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


app = FastAPI()

video_transcripts = {}

class URLRequest(BaseModel):
    url: str

class Message(BaseModel):
    role: str
    content: str

class MessageHistory(BaseModel):
    messages: list[Message]


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def get_youtube_transcript(youtube_url: str, lang="en") -> str:
    try:
        video_id = extract_video_id(youtube_url)
        print(f"Extracted video ID: {video_id}")
        output_file = f"{video_id}.{lang}.json3"

        # Try manually uploaded subtitles first
        manual_command = [
            "yt-dlp",
            "--skip-download",
            "--write-sub",
            "--sub-lang", lang,
            "--sub-format", "json3",
            "-o", "%(id)s.%(ext)s",
            youtube_url
        ]

        subprocess.run(manual_command)

        # Check if manual subtitle file exists
        if not os.path.exists(output_file):
            print("Manual subtitles not found. Trying auto-generated subtitles...")

            # Try auto-generated subtitles
            auto_command = [
                "yt-dlp",
                "--skip-download",
                "--write-auto-sub",
                "--sub-lang", lang,
                "--sub-format", "json3",
                "-o", "%(id)s.%(ext)s",
                youtube_url
            ]
            subprocess.run(auto_command, check=True)

        # If subtitle file exists now, parse it
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            transcript = ""
            for event in data.get("events", []):
                if "segs" in event:
                    transcript += "".join(seg.get("utf8", "") for seg in event["segs"]) + " "

            os.remove(output_file)
            return transcript.strip()
        else:
            raise FileNotFoundError("No subtitles found (manual or auto-generated).")

    except Exception as e:
        raise RuntimeError(f"Transcript extraction failed: {e}")


def convert_message(message: Message) -> BaseMessage:
    if message.role == "system":
        return SystemMessage(content=message.content)
    elif message.role == "user":
        return HumanMessage(content=message.content)
    elif message.role == "assistant":
        return AIMessage(content=message.content)
    else:
        raise ValueError(f"Unknown role: {message.role}")


model = ChatGroq(model="llama-3.1-8b-instant")

class QARequest(BaseModel):
    video_id: str
    query: str


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@app.post("/ask")
async def ask_question(req: QARequest):
    try:
        video_id_1 = extract_video_id(req.video_id)
        vector_path = f"faiss_stores/{video_id_1}"
        print(f"Loading vector store from: {vector_path}")
        print(f"faiss_stores/{req.video_id}")
        if not os.path.exists(vector_path):
            raise HTTPException(status_code=404, detail="Vector store not found for this video ID")

        db = FAISS.load_local(f"faiss_stores/{video_id_1}", embedding_model, allow_dangerous_deserialization=True)
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            chain_type="stuff"
        )

        answer = qa_chain.run(req.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@app.post("/transcript")
async def fetch_transcript(request: URLRequest):
    try:
        transcript = get_youtube_transcript(request.url)
        video_id = extract_video_id(request.url)

        # Split transcript into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.create_documents([transcript])

        # Store in FAISS
        vectorstore = FAISS.from_documents(docs, embedding_model)

        # Save vector store
        os.makedirs("faiss_stores", exist_ok=True)
        print(f"faiss_stores/{video_id}")
        vectorstore.save_local(f"faiss_stores/{video_id}")

        return {"message": f"Transcript stored in FAISS under ID '{video_id}'", "chunks": len(docs)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))