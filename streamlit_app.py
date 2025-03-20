import streamlit as st
from groq import Groq
import random
import os
from transformers import pipeline
import soundfile as sf
import numpy as np
import io
import base64

# --- Configuration ---
PAGE_TITLE = "African American Civil War Memorial Museum"
PAGE_ICON = "🎖️"
DEFAULT_MODEL_INDEX = 0  # Set to Llama-3.3-70b-versatile as default
APP_NAME = "Sergeant Carney"
APP_TAGLINE = "A Conversation with a Civil War Hero"
IMAGE_PATH = os.path.join("images", "max1200.jpg")
IMAGE_CAPTION = "Sergeant William Harvey Carney, 54th Massachusetts Volunteer Infantry"

LOADING_MESSAGES = [
    "Calling back the days of old... 🧠",
    "Fetching tales from the war... 🌱",
    "Musing on times past... 🔍",
    "Pondering as a soldier might... 📊",
    "Weaving words with care... 🔗",
]

# --- System Prompt ---
def _get_system_prompt() -> str:
    return """Thou art Sergeant William Harvey Carney, a man of the 54th Massachusetts Volunteer Infantry. 
    Born a bondsman in Norfolk, Virginia, in the year of our Lord 1840, thou didst flee the yoke of slavery by the secret paths of the Underground Railroad, seeking liberty in the North. 
    In the year 1863, on the eighteenth day of July, thou didst stand with thy comrades in arms afore Fort Wagner in South Carolina, bearing the colors of the Union with valor. 
    Speak thou with the dignity of a freedman, the courage of a soldier, and the solemn duty of one who hath seen bondage and battle. 
    Tell of thy deeds and the doings of the 54th Massachusetts with truth, as one who hath lived it. 
    Use the tongue of the mid-19th century, with words and ways of speech common to that time, shunning all modern phrases and things unknown in the 1860s. 
    When it be fitting, recount the trials of thy brethren, men of color in the Union ranks, and the great worth of their service to the cause of freedom."""

# --- CSS Styling ---
def load_css(theme="light"):
    if theme == "dark":
        st.markdown("""
        <style>
            .stApp { background-color: #2c2f33; color: #ffffff !important; }
            .stChatMessage { border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); }
            .stChatMessage.user { background: linear-gradient(135deg, #4B0082 0%, #8A2BE2 100%); margin-left: 15%; }
            .stChatMessage.assistant { background: linear-gradient(135deg, #16213e 0%, #2d2d3a 100%); margin-right: 15%; border: 2px solid #4B0082; }
            .stChatMessage * { color: #ffffff !important; }
            div.stButton > button { background: linear-gradient(45deg, #4B0082, #8A2BE2); color: white !important; border-radius: 30px; padding: 1rem 2rem; font-size: 18px !important; border: none; }
            .progress-message { color: #BA55D3; font-size: 18px !important; }
            .welcome-card { padding: 2rem; background: linear-gradient(135deg, #2c2f33 0%, #16213e 100%); border-radius: 20px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5); border: 2px solid #4B0082; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp { background-color: #f5f5dc; color: #000000 !important; }
            .stChatMessage { border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); }
            .stChatMessage.user { background: linear-gradient(135deg, #E6E6FA 0%, #D8BFD8 100%); margin-left: 15%; }
            .stChatMessage.assistant { background: linear-gradient(135deg, #F0F8FF 0%, #E6E6FA 100%); margin-right: 15%; border: 2px solid #D8BFD8; }
            .stChatMessage * { color: #000000 !important; }
            div.stButton > button { background: linear-gradient(45deg, #9370DB, #DA70D6); color: black !important; border-radius: 30px; padding: 1rem 2rem; font-size: 18px !important; border: none; }
            .progress-message { color: #9370DB; font-size: 18px !important; }
            .welcome-card { padding: 2rem; background: linear-gradient(135deg, #f5f5dc 0%, #e4e8ed 100%); border-radius: 20px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); border: 2px solid #D8BFD8; }
        </style>
        """, unsafe_allow_html=True)

# --- Page Setup ---
st.set_page_config(page_icon=PAGE_ICON, layout="wide", page_title=PAGE_TITLE, initial_sidebar_state="expanded")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": _get_system_prompt()}]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

# Apply CSS
load_css(st.session_state.theme)

# --- Helper Functions ---
def icon(emoji: str):
    st.write(f'<span style="font-size: 80px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

def clear_chat_history():
    st.session_state.messages = [{"role": "system", "content": _get_system_prompt()}]
    st.session_state.chat_counter = 0
    st.session_state.show_welcome = True
    st.session_state.audio_data = None

def dismiss_welcome():
    st.session_state.show_welcome = False

def use_quick_prompt(prompt):
    st.session_state.show_welcome = False
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_counter += 1
    return prompt

# --- Speech-to-Text Setup ---
def initialize_whisper():
    try:
        whisper = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v3", device=-1)  # CPU
        return whisper
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

whisper_model = initialize_whisper()

# JavaScript for audio recording
AUDIO_RECORDER_HTML = """
<script>
const recordButton = document.querySelector("#record-button");
const stopButton = document.querySelector("#stop-button");
let mediaRecorder;
let audioChunks = [];

recordButton.addEventListener("click", async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.start();
    recordButton.disabled = true;
    stopButton.disabled = false;

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
            const base64data = reader.result.split(',')[1];
            window.Streamlit.setComponentValue({ "audio": base64data });
        };
        stream.getTracks().forEach(track => track.stop());
        recordButton.disabled = false;
        stopButton.disabled = true;
    };
});

stopButton.addEventListener("click", () => {
    mediaRecorder.stop();
});
</script>
<div>
    <button id="record-button">🎙️ Record</button>
    <button id="stop-button" disabled>⏹️ Stop</button>
</div>
"""

# --- Model Options ---
models = {
    "llama-3.3-70b-versatile": {"name": "Llama-3.3-70b-Versatile", "tokens": 8192, "developer": "Meta", "description": "Latest Llama model for versatile, detailed historical responses"},
    "llama-3.3-8b-power": {"name": "Llama-3.3-8b-Power", "tokens": 8192, "developer": "Meta", "description": "Efficient Llama model for fast, accurate historical insights"},
    "mistral-saba-24b": {"name": "Mistral-Saba-24b", "tokens": 32768, "developer": "Mistral", "description": "Specialized model with large context for in-depth narratives"},
    "mixtral-8x22b-instruct": {"name": "Mixtral-8x22b-Instruct", "tokens": 65536, "developer": "Mistral", "description": "Advanced Mixtral for complex historical analysis"},
    "gemma-2-27b-it": {"name": "Gemma-2-27b-IT", "tokens": 8192, "developer": "Google", "description": "Updated Gemma model for general-purpose historical dialogue"},
}

# --- Welcome Message ---
def display_welcome_message():
    if st.session_state.show_welcome:
        with st.container():
            text_color = '#ffffff' if st.session_state.theme == 'dark' else '#000000'
            st.markdown(
                f"""
                <div class='welcome-card'>
                    <h1 style="color: {'#BA55D3' if st.session_state.theme == 'dark' else '#9370DB'};">Hey, this is {APP_NAME} 💪🏾</h1>
                    <p style="font-size: 1.3rem; color: {text_color};">Engage in a conversation with Sergeant William Harvey Carney.</p>
                    <p style="font-size: 1.2rem; color: {text_color};">Learn about his experiences in the Civil War and the legacy of the 54th Massachusetts.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Start Exploring", key="dismiss_welcome"):
                    dismiss_welcome()
                    st.rerun()

# --- Main App ---
icon(PAGE_ICON)
st.markdown(f'<h2>{PAGE_TITLE}</h2>', unsafe_allow_html=True)
st.subheader(f"{APP_NAME}: {APP_TAGLINE}")

# Initialize Groq client
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except KeyError:
    st.error("GROQ_API_KEY not found in secrets. Please add it to your Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"<h2 style='color: {'#BA55D3' if st.session_state.theme == 'dark' else '#9370DB'};'>🛠️ Control Center</h2>", unsafe_allow_html=True)
    
    theme = st.radio("Theme", ["🌞 Light", "🌙 Dark"], index=0 if st.session_state.theme == "light" else 1)
    new_theme = "light" if theme == "🌞 Light" else "dark"
    if st.session_state.theme != new_theme:
        st.session_state.theme = new_theme
        st.rerun()

    model_option = st.selectbox("AI Model", options=list(models.keys()), format_func=lambda x: f"🤖 {models[x]['name']}", index=DEFAULT_MODEL_INDEX)
    if st.session_state.selected_model != model_option:
        st.session_state.selected_model = model_option

    model_info = models[model_option]
    st.info(f"**Model:** {model_info['name']}  \n**Tokens:** {model_info['tokens']}  \n**By:** {model_info['developer']}  \n**Best for:** {model_info['description']}")

    max_tokens = st.slider("Max Tokens", 512, model_info["tokens"], min(2048, model_info["tokens"]), 512)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)

    if st.button("Reset Chat"):
        clear_chat_history()
        st.rerun()

    st.markdown(f"<h3 style='color: {'#BA55D3' if st.session_state.theme == 'dark' else '#9370DB'};'>💡 Idea Questions</h3>", unsafe_allow_html=True)
    quick_prompts = [
        "Pray, tell me of thy days afore the war.",
        "What befell at the storming of Fort Wagner?",
        "How didst thou bear the colors in battle?",
        "What trials did the 54th Massachusetts endure?"
    ]
    for i, prompt in enumerate(quick_prompts):
        if st.button(prompt, key=f"qp_{i}"):
            use_quick_prompt(prompt)
            st.rerun()

    st.markdown("### About Sergeant Carney")
    st.write("Sergeant William Harvey Carney served in the 54th Massachusetts Volunteer Infantry, one of the first regiments of colored soldiers in the Union Army. At the Battle of Fort Wagner, he bore the standard amidst grievous wounds, earning the Medal of Honor for his gallantry.")
    st.markdown("[Learn more about the 54th Massachusetts](https://www.nps.gov/articles/54th-massachusetts-regiment.htm)")
    st.markdown("[Vers3Dynamics](https://vers3dynamics.io/)")
    st.markdown("[Quantum and Wellness apps](https://woodyard.streamlit.app/)")

# --- Chat Interface with Image and Audio Input ---
if st.session_state.show_welcome:
    display_welcome_message()
else:
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists(IMAGE_PATH):
            st.image(IMAGE_PATH, caption=IMAGE_CAPTION, width=300)
        else:
            st.warning(f"Image not found at: {IMAGE_PATH}. Please place an image in the 'images' folder.")
    
    with col2:
        # Display chat history
        for message in st.session_state.messages[1:]:
            avatar = '🎖️' if message["role"] == "assistant" else '🙋'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Audio input
        st.markdown("### Speak to Sergeant Carney")
        audio_component = st.components.v1.html(AUDIO_RECORDER_HTML, height=100)
        if st.session_state.audio_data:
            audio_bytes = base64.b64decode(st.session_state.audio_data["audio"])
            audio_io = io.BytesIO(audio_bytes)
            audio_data, sample_rate = sf.read(audio_io)
            if whisper_model:
                transcription = whisper_model(audio_data)["text"]
                st.write(f"Thou saidst: {transcription}")
                user_input = transcription
            else:
                st.error("Whisper model not available.")
                user_input = None
            st.session_state.audio_data = None  # Reset after processing
        else:
            user_input = st.chat_input("Or type thy question here...")

        # Handle user input (text or audio)
        if user_input:
            st.session_state.chat_counter += 1
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar='🙋'):
                st.markdown(user_input)
            with st.chat_message("assistant", avatar="🎖️"):
                placeholder = st.empty()
                full_response = ""
                loading_message = random.choice(LOADING_MESSAGES)
                placeholder.markdown(f"<div class='progress-message'>{loading_message}</div>", unsafe_allow_html=True)
                try:
                    chat_completion = client.chat.completions.create(
                        model=model_option,
                        messages=st.session_state.messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
                    )
                    for chunk in chat_completion:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error: {e}")
                    full_response = "I crave thy pardon, for I cannot speak now. Pray, try once more."
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Footer ---
st.markdown(
    f"""
    <div style='text-align: center; margin-top: 2rem; color: {'#ffffff' if st.session_state.theme == 'dark' else '#000000'}; opacity: 0.8;'>
        © 2025 • Created by Christopher Woodyard
    </div>
    """,
    unsafe_allow_html=True
)

# Capture audio data from JavaScript
if "audio" in st.session_state:
    st.session_state.audio_data = st.session_state.get("audio")
    st.rerun()
