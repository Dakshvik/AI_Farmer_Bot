import os
import json
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
from typing import Dict, Any, List, Tuple

try:
    from audio_recorder_streamlit import audio_recorder
    _audrec_available = True
except ImportError:
    _audrec_available = False


# ----------------------------
# Load Knowledge Base
# ----------------------------
def load_kb(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_any_kb() -> Dict[str, Any]:
    kb: Dict[str, Any] = {}
    candidates = ["dataset.json", "diseases.json"]
    for name in candidates:
        if os.path.exists(name):
            try:
                data = load_kb(name)
                for k, v in data.items():
                    if k not in kb:
                        kb[k] = v
            except Exception as e:
                st.error(f"Error loading {name}: {e}")
    return kb


# ----------------------------
# KB Utilities
# ----------------------------
def flatten_kb(kb: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Return a list of (category, query, answer) from the KB."""
    items: List[Tuple[str, str, str]] = []
    for category, entries in kb.items():
        if isinstance(entries, list):
            for e in entries:
                q = str(e.get("query", "")).strip()
                a = str(e.get("answer", "")).strip()
                if q and a:
                    items.append((category, q, a))
    return items


# ----------------------------
# Text-to-Speech
# ----------------------------
def speak_text(text: str):
    try:
        tts = gTTS(text=text, lang="en")
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        # Encode audio to base64
        b64 = base64.b64encode(fp.read()).decode()
        md = f"""
            <audio autoplay controls>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"TTS error: {e}")


# ----------------------------
# Speech-to-Text
# ----------------------------
def recognize_speech_from_audio(audio_bytes) -> str:
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return ""


# ----------------------------
# Get Bot Response
# ----------------------------
def get_bot_response(user_input: str, kb: Dict[str, Any]) -> str:
    """Heuristic search over KB to find the best matching answer.

    Strategy:
    - If the input mentions a category or related keywords, prefer entries from that category.
    - Prefer entries whose query text appears in the input (substring match).
    - Fallback to any keyword overlap between input and entry query.
    - Provide a friendly fallback with suggestions if nothing matches well.
    """
    ui = user_input.lower().strip()
    if not ui:
        return "Please type a farming question to get advice."

    items = flatten_kb(kb)
    if not items:
        return "Knowledge base is empty. Please add entries to dataset.json."

    # Simple category keyword map to improve intent detection
    cat_keywords = {
        "fertilizer": ["fertilizer", "fertiliser", "urea", "dap", "npk", "manure", "nutrient"],
        "pests": ["pest", "insect", "worm", "bollworm", "ipm", "spray", "neem"],
        "irrigation": ["irrigation", "water", "watering", "drip", "furrow"],
        "weather": ["weather", "rain", "drought", "frost", "forecast", "temperature"],
        "soil": ["soil", "ph", "acidic", "alkaline", "salinity", "testing", "gypsum", "lime"],
        "market": ["price", "market", "msp", "sell", "mandi", "buyer", "enam"],
    }

    def has_any(hay: str, needles: List[str]) -> bool:
        return any(n in hay for n in needles)

    # Score items
    best_score = 0.0
    best_answer = None
    for category, query, answer in items:
        score = 0.0

        # Category mention boosts score
        if category.lower() in ui or has_any(ui, cat_keywords.get(category.lower(), [])):
            score += 1.0

        ql = query.lower()

        # Direct substring match gets a strong boost
        if ql in ui or any(word in ui for word in ql.split() if len(word) > 2):
            score += 2.0

        # Light overlap: any UI token in query
        ui_tokens = [t for t in ui.replace("/", " ").replace(",", " ").split() if len(t) > 2]
        if any(t in ql for t in ui_tokens):
            score += 0.5

        if score > best_score:
            best_score = score
            best_answer = answer

    if best_answer and best_score >= 1.0:
        return best_answer

    # Friendly fallback with examples derived from KB categories
    tips = (
        "Try including the crop and topic. For example: "
        "'fertilizer for rice', 'pest control in tomato', 'irrigation for wheat', 'soil testing advice'."
    )
    return f"I don't have enough information to answer that precisely. {tips}"


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Farming Assistant", layout="wide")

st.sidebar.title("Settings")
enable_voice = st.sidebar.checkbox("Enable Voice", value=True)


st.sidebar.markdown("---")
st.sidebar.title("About")
st.sidebar.write("""
Ask me about:
- Fertilizer guidance  
- Pest and disease management  
- Irrigation schedules  
- Soil health tips  
- Weather-based advice  
- Market prices  
""")


st.title("🌱 Farming Assistant")
st.caption("Your personal farming assistant")

# Load KB
kb = load_any_kb()

# Conversation State
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Intro message
if not st.session_state.conversation:
    st.session_state.conversation.append({"role": "bot", "text": "Hello! I'm your farming assistant. How can I help you today?"})

# Conversation will be rendered after handling inputs


# ----------------------------
# Input Section
# ----------------------------
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    user_input = st.text_input("Type your farming question here...", key="user_text_input")

with col2:
    audio_bytes = None
    if _audrec_available:
        # Keep a stable key so the mic doesn't disappear on reruns
        audio_bytes = audio_recorder(key="input_mic")
    else:
        st.button("🎤", disabled=True, help="Install audio_recorder_streamlit for mic input")

with col3:
    send_clicked = st.button("Send", key="send_btn")


# Handle text input
if send_clicked and user_input:
    st.session_state.conversation.append({"role": "user", "text": user_input})
    response = get_bot_response(user_input, kb)
    st.session_state.conversation.append({"role": "bot", "text": response})

    # Speak response
    if enable_voice:
        speak_text(response)


# Handle mic input inline (beside the text input)
if _audrec_available and audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    text_from_speech = recognize_speech_from_audio(audio_bytes)
    if text_from_speech:
        st.session_state.conversation.append({"role": "user", "text": text_from_speech})
        response = get_bot_response(text_from_speech, kb)
        st.session_state.conversation.append({"role": "bot", "text": response})

        # Speak response
        if enable_voice:
            speak_text(response)
elif not _audrec_available:
    st.caption("Voice input unavailable: install `audio_recorder_streamlit`.")

# ----------------------------
# Render conversation (clean chat layout)
# ----------------------------
for msg in st.session_state.conversation:
    role = msg.get("role", "bot")
    if role == "user":
        with st.chat_message("user"):
            st.write(msg.get("text", ""))
    else:
        with st.chat_message("assistant"):
            st.write(msg.get("text", ""))
