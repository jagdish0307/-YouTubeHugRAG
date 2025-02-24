# 🎬 Audio-Enhanced Conversational AI Chatbot

## 📌 Project Overview
This Streamlit-based chatbot allows users to interact with YouTube videos in an entirely new way. By simply providing a YouTube link, users can extract audio, transcribe the content, and ask questions to receive accurate answers from the video transcript in real-time.

## 🔧 Features
- 🎥 Extract audio from YouTube videos
- ✍️ Transcribe audio using AssemblyAI
- 🤖 Ask questions and get answers using Hugging Face's API
- 🖥️ User-friendly Streamlit dashboard

## 🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd <project-folder>
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv env
```

### 3️⃣ Activate the Virtual Environment
- **Windows:**
  ```bash
  env\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source env/bin/activate
  ```

### 4️⃣ Install Project Requirements
```bash
pip install -r requirements.txt
```

### 5️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and add the following keys:

```
ASSEMBLY_AI_KEY=your_assemblyai_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### 6️⃣ Install FFMPEG (For Audio Processing)
- Add FFMPEG to your system path. Example for Windows:
```bash
$env:Path += ";C:\\Users\\<your-username>\\ffmpeg"
```

### 7️⃣ Run the Project
```bash
streamlit run app.py
```

---

## 💻 Usage Guide
1. Paste a YouTube video link into the input box.
2. Wait for the audio to be extracted and transcribed.
3. View the transcript in the app.
4. if you wants summery of this vidio click on Generate Summery
5. Ask any question related to the video and receive an answer in real-time.

---

## 📂 Folder Structure
```
📁 project-folder
│   📄 app.py              # Main Streamlit app file
│   📄 requirements.txt    # List of dependencies
│   📄 .env                # Environment variables
│   📄 README.md           # Project documentation
```

---

## 🛠️ Technologies Used
- [Streamlit](https://streamlit.io/) - For creating the interactive dashboard
- [Pytube](https://pytube.io/en/latest/) - To download YouTube audio
- [AssemblyAI](https://www.assemblyai.com/) - For audio transcription
- [Hugging Face](https://huggingface.co/) - For answering user queries

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙌 Acknowledgements
- Inspired by the need for efficient information retrieval from long-form audio and video content.

---

