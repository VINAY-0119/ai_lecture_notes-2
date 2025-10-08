import streamlit as st
import whisper
import tempfile
import openai
import os

# Load OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Lecture Voice to Notes Generator")

# Upload audio file
uploaded_file = st.file_uploader(
    "Upload Lecture Audio (wav, mp3, m4a)", type=["wav", "mp3", "m4a"]
)

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    st.info("Transcribing audio... This may take a few minutes depending on length.")

    # Load Whisper model
    model = whisper.load_model("small")  # small for speed; medium/large for better accuracy
    result = model.transcribe(audio_path)

    st.subheader("Transcribed Text")
    st.write(result['text'])

    # Generate summarized notes using GPT
    if st.button("Generate Notes"):
        st.info("Generating summarized notes...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that converts lecture transcripts into concise study notes, keywords, quizzes, and flashcards."},
                {"role": "user", "content": result['text']}
            ],
            max_tokens=1000
        )
        notes = response['choices'][0]['message']['content']
        st.subheader("Lecture Notes, Keywords, Quizzes, Flashcards")
        st.write(notes)
