import streamlit as st
import json
import os
import time
import sys
from dotenv import load_dotenv
import requests
import yt_dlp
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
api_token = os.getenv('ASSEMBLY_AI_KEY')
groq_token = os.getenv('GROQ_API_KEY')
openai_token = os.getenv('OPENAI_API_KEY')
huggingface_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

base_url = "https://api.assemblyai.com/v2"
headers = {
    "authorization": api_token,
    "content-type": "application/json"
}

# yt-dlp function for YouTube video
def save_audio(url):
    try:
        os.makedirs('temp', exist_ok=True)
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'temp/%(title)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_filename = ydl.prepare_filename(info).replace('.webm', '.mp3')
        logger.info(f"Successfully downloaded audio: {audio_filename}")
        return Path(audio_filename).name
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        st.error(f"Error downloading audio: {str(e)}")
        return None

# AssemblyAI transcription
def assemblyai_stt(audio_filename):
    try:
        audio_path = os.path.join('temp', audio_filename)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        with open(audio_path, "rb") as f:
            response = requests.post(base_url + "/upload", headers=headers, data=f)
        response.raise_for_status()
        upload_url = response.json()["upload_url"]
        data = {"audio_url": upload_url}
        url = base_url + "/transcript"
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        transcript_id = response.json()['id']
        polling_endpoint = f"{base_url}/transcript/{transcript_id}"
        while True:
            transcription_result = requests.get(polling_endpoint, headers=headers).json()
            if transcription_result['status'] == 'completed':
                break
            elif transcription_result['status'] == 'error':
                raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
            else:
                time.sleep(3)
        transcription_text = transcription_result['text']
        word_timestamps = transcription_result['words']
        os.makedirs('docs', exist_ok=True)
        with open('docs/transcription.txt', 'w') as file:
            file.write(transcription_text)
        with open('docs/word_timestamps.json', 'w') as file:
            json.dump(word_timestamps, file)
        logger.info("Successfully transcribed audio with word-level timestamps")
        return transcription_text, word_timestamps
    except Exception as e:
        logger.error(f"Error in speech-to-text conversion: {str(e)}")
        st.error(f"Error in speech-to-text conversion: {str(e)}")
        return None, None
# Set up QA Chain using Hugging Face
def setup_qa_chain():
    try:
        loader = TextLoader('docs/transcription.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever()

        # Use a model that supports text2text-generation for QA
        chat = HuggingFaceHub(
            repo_id="google/flan-t5-base", 
            task="text2text-generation", 
            huggingfacehub_api_token=huggingface_token
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        with open('docs/word_timestamps.json', 'r') as file:
            word_timestamps = json.load(file)
        return qa_chain, word_timestamps

    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        st.error(f"Error setting up QA chain: {str(e)}")
        return None, None


# Summary generation using Hugging Face
def generate_summary(transcription):
    # Use a model designed for summarization
    chat = HuggingFaceHub(
        repo_id="facebook/bart-large-cnn", 
        task="summarization", 
        huggingfacehub_api_token=huggingface_token
    )

    summary_prompt = PromptTemplate(
        input_variables=["transcription"],
        template="Summarize the following transcription in 3-5 sentences:\n\n{transcription}"
    )
    summary_chain = LLMChain(llm=chat, prompt=summary_prompt)
    summary = summary_chain.run(transcription)
    return summary

# Main Streamlit app
st.set_page_config(layout="wide", page_title="ChatAudio - Hugging Face", page_icon="🔊")
st.title("Chat with Your Audio using Hugging Face LLM")

input_source = st.text_input("Enter the YouTube video URL")
if input_source:
    col1, col2 = st.columns(2)
    with col1:
        st.info("Your uploaded video")
        st.video(input_source)
        audio_filename = save_audio(input_source)
        if audio_filename:
            transcription, word_timestamps = assemblyai_stt(audio_filename)
            if transcription:
                st.info("Transcription completed. You can now ask questions.")
                st.text_area("Transcription", transcription, height=300)
                qa_chain, word_timestamps = setup_qa_chain()
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcription)
                        st.subheader("Summary")
                        st.write(summary)
    with col2:
        st.info("Chat Below")
        query = st.text_input("Ask your question here...")
        if query:
            if qa_chain:
                with st.spinner("Generating answer..."):
                    result = qa_chain({"query": query})
                    answer = result['result']
                    st.success(answer)
            else:
                st.error("QA system is not ready. Please make sure the transcription is completed.")
