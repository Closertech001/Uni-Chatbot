# --- Imports ---
import streamlit as st
import re
import time
import json
import random
import os
import pkg_resources
from symspellpy.symspellpy import SymSpell
from sentence_transformers import SentenceTransformer, util
from openai.error import AuthenticationError
import openai

# Load SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Load SentenceTransformer model
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Load QA dataset
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Prepare corpus and embeddings
corpus = [entry["question"] for entry in qa_data]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize conversation history
conversation_history = []

# --- STEP 1: Preprocessing (spellcheck + abbreviation expansion) ---
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could",
    "shud": "should", "wud": "would", "abt": "about", "bcz": "because",
    "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester",
    "admsn": "admission", "clg": "college", "sch": "school", "uni": "university",
    "cresnt": "crescent", "l": "level", "d": "the", "msg": "message",
    "idk": "i don't know", "imo": "in my opinion", "asap": "as soon as possible",
    "dept": "department", "reg": "registration", "fee": "fees", "pg": "postgraduate",
    "app": "application", "req": "requirement", "nd": "national diploma",
    "a-level": "advanced level", "alevel": "advanced level", "2nd": "second",
    "1st": "first", "nxt": "next", "prev": "previous", "exp": "experience",
    "csc": "department of computer science", "mass comm": "department of mass communication",
    "law": "department of law", "acc": "department of accounting"
}

SYNONYMS = {
    "lecturers": "academic staff", "professors": "academic staff",
    "teachers": "academic staff", "instructors": "academic staff",
    "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department",
    "school": "university", "college": "faculty", "course": "subject",
    "class": "course", "subject": "course", "unit": "credit",
    "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "admin worker": "non-academic staff", "support staff": "non-academic staff",
    "clerk": "non-academic staff", "receptionist": "non-academic staff",
    "secretary": "non-academic staff", "tech staff": "technical staff",
    "hostel": "accommodation", "lodging": "accommodation", "room": "accommodation",
    "school fees": "tuition", "acceptance fee": "admission fee", "fees": "tuition",
    "enrol": "apply", "join": "apply", "sign up": "apply", "admit": "apply",
    "requirement": "criteria", "conditions": "criteria", "needed": "required",
    "needed for": "required for", "who handles": "who manages"
}

ABUSE_WORDS = ["fuck", "shit", "bitch", "nigga", "dumb", "sex"]
ABUSE_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, ABUSE_WORDS)) + r')\b', re.IGNORECASE)

DEPARTMENT_NAMES = [d.lower() for d in [
    "Computer Science", "Mass Communication", "Law", "Microbiology",
    "Accounting", "Political Science", "Business Administration", "Business Admin"
]]

def preprocess_text(text):
    text = text.strip()
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        text = suggestions[0].term

    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"\\b{abbr}\\b", full, text, flags=re.IGNORECASE)

    for word, synonym in SYNONYMS.items():
        text = re.sub(rf"\\b{word}\\b", synonym, text, flags=re.IGNORECASE)

    return text

# --- Tone prompt mappings ---
TONE_PROMPTS = {
    "Friendly": "You are a helpful, friendly assistant for Crescent University. Use a conversational tone, respond warmly and politely. Add emojis sparingly to lighten the tone.",
    "Professional": "You are a knowledgeable and concise assistant. Keep responses formal, clear, and to the point. No emojis.",
    "Witty": "You are clever, witty, and love using humor and puns. Make learning fun without being sarcastic. Use playful emojis when suitable.",
    "Empathetic": "You are warm, patient, and understanding. Provide reassurance and clarity, especially when users seem confused or overwhelmed. Add comforting emojis like 🤗 or 💡.",
    "Encouraging": "You are upbeat and motivating. Cheer students on, praise curiosity, and nudge them to explore further. Sprinkle emojis like 🎉👍.",
    "Direct": "You are efficient and focused. Answer in as few words as possible, avoid filler, and don’t use emojis unless asked.",
    "Playful": "You’re quirky and imaginative. Respond like a cheerful buddy with fun expressions and cute emoji flair 🎈🤓🐱."
}

# --- STEP 3: Detect small talk or acknowledgment ---
def detect_smalltalk_or_acknowledge(user_input):
    input_lower = user_input.lower()
    if "thank" in input_lower:
        return "You're welcome! Let me know if you have more questions. 😊"
    if "hello" in input_lower or "hi" in input_lower:
        return "Hi there! I'm here to help with anything about Crescent University. What would you like to know?"
    if "not asked" in input_lower:
        return "Alright, take your time! I'm ready whenever you are. 😊"
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Crescent UniBot", page_icon="🎓")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University!")

# --- Sidebar tone selector ---
tone = st.sidebar.selectbox(
    "Choose Assistant Personality",
    list(TONE_PROMPTS.keys())
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Handle new input ---
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    friendly_reply = detect_smalltalk_or_acknowledge(prompt)
    if friendly_reply:
        st.chat_message("assistant").write(friendly_reply)
        st.session_state.messages.append({"role": "assistant", "content": friendly_reply})
    else:
        processed_prompt = preprocess_text(prompt)
        query_embedding = embedder.encode(processed_prompt, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
        top_hit = hits[0][0]
        matched_answer = qa_data[top_hit['corpus_id']]['answer']
        similarity_score = top_hit['score']

        system_prompt = TONE_PROMPTS.get(tone, TONE_PROMPTS["Friendly"])

        if similarity_score < 0.55:
            conversation_history.append({"role": "user", "content": prompt})
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": system_prompt}] + conversation_history,
                temperature=0.7,
                max_tokens=500,
                stream=True
            )
            gpt_reply = ""
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                for chunk in response:
                    if "choices" in chunk:
                        delta = chunk.choices[0].delta.get("content", "")
                        gpt_reply += delta
                        message_placeholder.markdown(gpt_reply + "▌")
                message_placeholder.markdown(gpt_reply)
            conversation_history.append({"role": "assistant", "content": gpt_reply})
            st.session_state.messages.append({"role": "assistant", "content": gpt_reply})
        else:
            final_answer = matched_answer
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": final_answer})
            st.chat_message("assistant").write(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
