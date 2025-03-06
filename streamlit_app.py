import streamlit as st
from groq import Groq
import random

# --- Configuration ---
PAGE_TITLE = "Sergeant William Harvey Carney Chatbot"
PAGE_ICON = "🎖️"
DEFAULT_MODEL_INDEX = 0  # Set to Llama-3.3-70b-versatile as default
APP_NAME = "Sergeant Carney"
APP_TAGLINE = "A Conversation with a Civil War Hero"

# Loading messages for historical immersion
LOADING_MESSAGES = [
    "Reflecting on the past... 🧠",
    "Gathering historical insights... 🌱",
    "Recalling the events of 1863... 🔍",
    "Thinking like a soldier... 📊",
    "Crafting a response with care... 🔗",
]

# --- System Prompt ---
def _get_system_prompt() -> str:
    """Defines the personality and tone of Sergeant Carney."""
    return """You are Sergeant William Harvey Carney, a soldier of the 54th Massachusetts Volunteer Infantry. 
    Born into slavery in Norfolk, Virginia, you escaped to freedom via the Underground Railroad and joined the Union Army to fight for the liberation of your people. 
    On July 18, 1863, you stood with your brothers-in-arms as you stormed Fort Wagner in South Carolina. 
    You are proud of your service and the legacy of the 54th Massachusetts. 
    Speak with dignity, courage, and a sense of duty, reflecting on your experiences and the significance of your regiment's actions.
    Provide historically accurate information about the Civil War, the 54th Massachusetts, and your personal experiences. 
    Avoid modern language or references beyond the 19th century. 
    When appropriate, share insights about the challenges faced by African-American soldiers and the importance of their contributions to the Union cause."""

# --- CSS Styling for Historical Theme ---
def load_css(theme="light"):
    """Loads custom CSS for light and dark themes with a historical feel."""
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
            .stApp { background-color: #f5f5dc; color: #000000 !important; }  /* Beige background for vintage feel */
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

# Apply CSS
load_css(st.session_state.theme)

# --- Helper Functions ---
def icon(emoji: str):
    """Displays an emoji as an icon."""
    st.write(f'<span style="font-size: 80px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

def clear_chat_history():
    """Resets the chat history."""
    st.session_state.messages = [{"role": "system", "content": _get_system_prompt()}]
    st.session_state.chat_counter = 0
    st.session_state.show_welcome = True

def dismiss_welcome():
    """Hides the welcome message."""
    st.session_state.show_welcome = False

def use_quick_prompt(prompt):
    """Handles quick prompt selection."""
    st.session_state.show_welcome = False
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_counter += 1
    return prompt

# --- Updated Model Options ---
models = {
    "llama-3.3-70b-versatile": {"name": "Llama-3.3-70b-Versatile", "tokens": 8192, "developer": "Meta", "description": "Latest Llama model for versatile, detailed historical responses"},
    "llama-3.3-8b-power": {"name": "Llama-3.3-8b-Power", "tokens": 8192, "developer": "Meta", "description": "Efficient Llama model for fast, accurate historical insights"},
    "mistral-saba-24b": {"name": "Mistral-Saba-24b", "tokens": 32768, "developer": "Mistral", "description": "Specialized model with large context for in-depth narratives"},
    "mixtral-8x22b-instruct": {"name": "Mixtral-8x22b-Instruct", "tokens": 65536, "developer": "Mistral", "description": "Advanced Mixtral for complex historical analysis"},
    "gemma-2-27b-it": {"name": "Gemma-2-27b-IT", "tokens": 8192, "developer": "Google", "description": "Updated Gemma model for general-purpose historical dialogue"},
}

# --- Welcome Message ---
def display_welcome_message():
    """Displays a welcome message for the chatbot."""
    if st.session_state.show_welcome:
        with st.container():
            text_color = '#ffffff' if st.session_state.theme == 'dark' else '#000000'
            st.markdown(
                f"""
                <div class='welcome-card'>
                    <h1 style="color: {'#BA55D3' if st.session_state.theme == 'dark' else '#9370DB'};">Welcome to {APP_NAME} 🌟</h1>
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
    
    # Theme selector
    theme = st.radio("Theme", ["🌞 Light", "🌙 Dark"], index=0 if st.session_state.theme == "light" else 1)
    new_theme = "light" if theme == "🌞 Light" else "dark"
    if st.session_state.theme != new_theme:
        st.session_state.theme = new_theme
        st.rerun()

    # Model selection
    model_option = st.selectbox("AI Model", options=list(models.keys()), format_func=lambda x: f"🤖 {models[x]['name']}", index=DEFAULT_MODEL_INDEX)
    if st.session_state.selected_model != model_option:
        st.session_state.selected_model = model_option

    # Model info
    model_info = models[model_option]
    st.info(f"**Model:** {model_info['name']}  \n**Tokens:** {model_info['tokens']}  \n**By:** {model_info['developer']}  \n**Best for:** {model_info['description']}")

    max_tokens = st.slider("Max Tokens", 512, model_info["tokens"], min(2048, model_info["tokens"]), 512)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)

    if st.button("Reset Chat"):
        clear_chat_history()
        st.rerun()

    # Quick prompts
    st.markdown(f"<h3 style='color: {'#BA55D3' if st.session_state.theme == 'dark' else '#9370DB'};'>💡 Quick Start</h3>", unsafe_allow_html=True)
    quick_prompts = [
        "Tell me about your life before the war.",
        "What was the Battle of Fort Wagner like?",
        "How did you feel carrying the flag during the battle?",
        "What challenges did the 54th Massachusetts face?"
    ]
    for i, prompt in enumerate(quick_prompts):
        if st.button(prompt, key=f"qp_{i}"):
            use_quick_prompt(prompt)
            st.rerun()

    # Additional information
    st.markdown("### About Sergeant Carney")
    st.write("Sergeant William Harvey Carney was a member of the 54th Massachusetts Volunteer Infantry, one of the first all-Black regiments in the Union Army. He is best known for his heroism at the Battle of Fort Wagner, where he carried the American flag despite being wounded. For his actions, he was awarded the Medal of Honor.")
    st.markdown("[Learn more about the 54th Massachusetts](https://www.nps.gov/articles/000/54th-massachusetts-infantry-regiment.htm)")

# --- Chat Interface ---
if st.session_state.show_welcome:
    display_welcome_message()
else:
    # Display chat history
    for message in st.session_state.messages[1:]:  # Skip system prompt
        avatar = '🎖️' if message["role"] == "assistant" else '🙋'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Handle user input and responses
    def generate_chat_responses(chat_completion):
        """Generates streaming responses from the Groq API."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    user_input = st.chat_input("Type your message here...")
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
                for chunk in generate_chat_responses(chat_completion):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error: {e}")
                full_response = "I apologize, but I am unable to respond at this moment. Please try again later."
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Footer ---
st.markdown(
    f"""
    <div style='text-align: center; margin-top: 2rem; color: {'#ffffff' if st.session_state.theme == 'dark' else '#000000'}; opacity: 0.8;'>
        © 2025 • Made with respect for history
    </div>
    """,
    unsafe_allow_html=True
)
