import os
import json
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
from typing import Dict, Any, List, Tuple

# Import the Hugging Face client
from huggingface_hub import InferenceClient

# This component is optional but provides a better UX for audio recording
try:
    from audio_recorder_streamlit import audio_recorder
    _audrec_available = True
except ImportError:
    _audrec_available = False

# ----------------------------
# Knowledge Base (KB) Functions
# ----------------------------
@st.cache_data(show_spinner="Loading knowledge base...")
def load_any_kb() -> Dict[str, Any]:
    """Loads and merges knowledge base files."""
    kb: Dict[str, Any] = {}
    candidates = ["dataset.json", "diseases.json"]
    for name in candidates:
        if os.path.exists(name):
            try:
                with open(name, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Merge data: new keys are added, existing ones are ignored
                for k, v in data.items():
                    if k not in kb:
                        kb[k] = v
            except Exception as e:
                st.error(f"Error loading or parsing {name}: {e}")
    return kb

def flatten_kb(kb: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Converts the nested KB dictionary into a flat list of (category, query, answer)."""
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
# Voice and Speech Functions
# ----------------------------
def speak_text_autoplay(text: str):
    """Generates speech and plays it automatically in the browser using a hidden audio element."""
    try:
        tts = gTTS(text=text, lang="en")
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        b64 = base64.b64encode(fp.read()).decode()
        # HTML for an audio player that autoplays and is hidden
        md = f"""
            <audio autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred with text-to-speech: {e}")

def recognize_speech_from_audio(audio_bytes: bytes) -> str:
    """Transcribes audio bytes into text using Google's speech recognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        st.warning("Could not understand the audio. Please try again.")
        return ""
    except sr.RequestError as e:
        st.error(f"Speech recognition service error; {e}")
        return ""
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return ""

# ----------------------------
# Bot Response Logic
# ----------------------------
def get_bot_response(user_input: str, kb_items: List[Tuple[str, str, str]]) -> str:
    """Finds the best matching factual answer from the KB using a scoring heuristic."""
    ui = user_input.lower().strip()
    if not ui or not kb_items:
        return "" # Return empty if no input or no KB

    cat_keywords = {
        "fertilizer": ["fertilizer", "manure", "nutrient"], "pests": ["pest", "insect", "worm", "spray"],
        "irrigation": ["irrigation", "water", "watering"], "weather": ["weather", "rain", "forecast"],
        "soil": ["soil", "ph", "testing"], "market": ["price", "market", "msp", "sell"],
    }
    best_score = 0.0
    best_answer = None
    for category, query, answer in kb_items:
        score = 0.0
        ql = query.lower()
        if any(keyword in ui for keyword in cat_keywords.get(category, [])): score += 1.0
        if ql in ui: score += 2.0
        ui_tokens, query_tokens = set(ui.split()), set(ql.split())
        overlap = len(ui_tokens.intersection(query_tokens))
        score += 0.5 * overlap
        if score > best_score:
            best_score, best_answer = score, answer
            
    return best_answer if best_score >= 1.0 else ""

def get_hf_response(user_input: str, kb_answer: str) -> str:
    """Uses a Hugging Face model to make the KB answer sound more natural."""
    fallback_message = "I'm sorry, I couldn't find a specific answer for that. Please try rephrasing your question."
    if not kb_answer:
        return fallback_message

    if "HF_TOKEN" not in st.secrets:
        st.error("Hugging Face API token not found. Please add it to your Streamlit secrets.")
        return kb_answer # Fallback to the basic answer

    try:
        client = InferenceClient(token=st.secrets["HF_TOKEN"])
        prompt = f"""
        You are a friendly farming assistant. Your task is to answer the user's question in a conversational and helpful way, based on the provided information.
        Make the answer easy to understand and keep it concise (2-3 sentences).

        User's Question: "{user_input}"
        Information to Use for the Answer: "{kb_answer}"

        Your Friendly Answer:
        """
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="HuggingFaceH4/zephyr-7b-beta",
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Could not connect to Hugging Face, providing the standard answer. Error: {e}")
        return kb_answer

# ----------------------------
# Main Streamlit App UI
# ----------------------------
def main():
    st.set_page_config(page_title="Farming Assistant", page_icon="🌱", layout="centered")

    st.title("🌱 Farming Assistant")
    st.caption("Your AI-powered guide for farming questions.")

    with st.sidebar:
        st.header("Settings")
        enable_voice = st.checkbox("Enable Voice Output", value=True)
        st.markdown("---")
        
        st.header("About")
        st.write(
            "This assistant answers questions on farming topics "
            "using an AI model."
        )

        st.markdown("---")
        st.subheader("Things You Can Ask:")
        st.info("What is the best fertilizer for rice?")
        st.info("How do I control pests in tomato plants?")
        st.info("What is the market price for cotton?")
        st.info("How can I test my soil's pH level?")
        
        # This is the helper text you requested
        st.markdown("---")
        st.caption("💡 In case of any errors, please refresh the page.")

    kb, kb_items = load_any_kb(), flatten_kb(load_any_kb())

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your farming today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = None
    input_col, mic_col = st.columns([4, 1])

    with mic_col:
        st.write("")
        st.write("")
        if _audrec_available:
            audio_bytes = audio_recorder(text="🎤", icon_size="2xl", key="audio_recorder")
            if audio_bytes:
                with st.spinner("Transcribing your voice..."):
                    prompt = recognize_speech_from_audio(audio_bytes)
        else:
            st.info("Install `audio-recorder-streamlit` for voice input.")

    with input_col:
        if text_input := st.chat_input("Ask your farming question here..."):
            prompt = text_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                basic_answer = get_bot_response(prompt, kb_items)
                final_response = get_hf_response(prompt, basic_answer)
            st.markdown(final_response)
        
        st.session_state.messages.append({"role": "assistant", "content": final_response})

        if enable_voice:
            speak_text_autoplay(final_response)
if __name__ == "__main__":
    main()
