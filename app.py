import streamlit as st
import whisper
import tempfile
import openai
import requests
import time

# --------------------------
# Page configuration
# --------------------------
st.set_page_config(
    page_title="Lecture Notes Generator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Sidebar: Theme toggle and options
# --------------------------
st.sidebar.title("Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
st.markdown(f"""<style>
body {{background-color: {"#ffffff" if theme=="Light" else "#0e1117"}; color: {"#000000" if theme=="Light" else "#f0f0f0"};}}
</style>""", unsafe_allow_html=True)

st.sidebar.markdown("### Options")
language = st.sidebar.selectbox("Output Language", ["English", "Spanish", "French", "German", "Hindi"])
generate_notes = st.sidebar.checkbox("Generate Notes", True)
generate_mcq = st.sidebar.checkbox("Generate MCQs", True)
generate_flashcards = st.sidebar.checkbox("Generate Flashcards / Keywords", True)

# --------------------------
# Load API keys from Streamlit Secrets
# --------------------------
openai_api_key = st.secrets.get("OPENAI_API_KEY")
gemini_api_key = st.secrets.get("GEMINI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found in Streamlit Secrets!")
    st.stop()
if not gemini_api_key:
    st.error("Gemini API key not found in Streamlit Secrets!")
    st.stop()

openai.api_key = openai_api_key

# --------------------------
# App title and description
# --------------------------
st.title("🎙️ Lecture Notes Generator")
st.markdown("""
Upload your lecture audio to generate:
- **Transcribed text**
- **Concise study notes**
- **MCQs**
- **Flashcards / Keywords**
Supports multiple languages!
""")

# --------------------------
# Upload audio file
# --------------------------
uploaded_file = st.file_uploader(
    "Upload Lecture Audio (wav, mp3, m4a)",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    # --------------------------
    # Transcription with progress bar
    # --------------------------
    st.info("Transcribing audio with Whisper...")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    model = whisper.load_model("small")
    # simulate progress
    for i in range(0, 101, 10):
        time.sleep(0.1)
        progress_text.text(f"Transcribing... {i}%")
        progress_bar.progress(i)

    result = model.transcribe(audio_path)

    st.subheader("📝 Transcribed Text")
    st.text_area("Transcript", value=result['text'], height=200)

    # --------------------------
    # Tabs for outputs
    # --------------------------
    tabs = st.tabs(["Notes", "MCQs", "Flashcards / Keywords", "Gemini API Test"])

    # --------------------------
    # Function to generate AI content
    # --------------------------
    def generate_gpt_output(prompt, max_tokens=1000):
        with st.spinner("Generating AI content..."):
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": result['text']}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

    # --------------------------
    # Generate Notes
    # --------------------------
    if generate_notes:
        with tabs[0]:
            try:
                prompt_notes = f"You are an assistant that converts lecture transcripts into concise, structured study notes in {language}."
                notes = generate_gpt_output(prompt_notes)
                st.text_area("Lecture Notes", value=notes, height=300)
                st.download_button("Download Notes", notes, file_name="lecture_notes.txt", mime="text/plain")
            except Exception as e:
                st.error(f"Error generating notes: {e}")

    # --------------------------
    # Generate MCQs
    # --------------------------
    if generate_mcq:
        with tabs[1]:
            try:
                prompt_mcq = f"You are an assistant that generates 5 multiple-choice questions from a lecture transcript in {language}."
                mcqs = generate_gpt_output(prompt_mcq, max_tokens=500)
                st.text_area("MCQs", value=mcqs, height=300)
                st.download_button("Download MCQs", mcqs, file_name="lecture_mcqs.txt", mime="text/plain")
            except Exception as e:
                st.error(f"Error generating MCQs: {e}")

    # --------------------------
    # Generate Flashcards / Keywords
    # --------------------------
    if generate_flashcards:
        with tabs[2]:
            try:
                prompt_flash = f"You are an assistant that extracts key points, keywords, and creates flashcards from a lecture transcript in {language}."
                flashcards = generate_gpt_output(prompt_flash, max_tokens=500)
                st.text_area("Flashcards / Keywords", value=flashcards, height=300)
                st.download_button("Download Flashcards", flashcards, file_name="lecture_flashcards.txt", mime="text/plain")
            except Exception as e:
                st.error(f"Error generating flashcards: {e}")

    # --------------------------
    # Gemini API Test
    # --------------------------
    with tabs[3]:
        st.info("Testing Gemini API...")
        try:
            url = "https://api.gemini.com/v1/some_endpoint"  # Replace with actual endpoint
            headers = {"Authorization": f"Bearer {gemini_api_key}"}
            r = requests.get(url, headers=headers)
            st.json(r.json())
        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, OpenAI, Whisper, and Gemini API.")
