# --- Imports ---
import streamlit as st
import re
import json
import time
from symspellpy.symspellpy import SymSpell
from sentence_transformers import SentenceTransformer, util
import openai

# Load SymSpell for spell correction (only once)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Load SentenceTransformer model (only once)
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Load QA dataset (only once)
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Prepare corpus and embeddings (only once)
corpus = [entry["question"] for entry in qa_data]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize conversation history for fallback GPT
conversation_history = []

# --- Preprocessing dicts ---
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

# --- Preprocessing function ---
def preprocess_text(text):
    text = text.strip()

    # Skip spellcheck for short inputs (speed up)
    if len(text.split()) >= 4:
        suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            text = suggestions[0].term

    # Replace abbreviations
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"\\b{abbr}\\b", full, text, flags=re.IGNORECASE)

    # Replace synonyms
    for word, synonym in SYNONYMS.items():
        text = re.sub(rf"\\b{word}\\b", synonym, text, flags=re.IGNORECASE)

    return text

# --- Friendly tone control ---
def add_human_tone(raw_response):
    friendly_prefix = "Of course! 😊 "
    if raw_response.lower().startswith("sorry") or "i don't know" in raw_response.lower():
        return "Hmm, let me double-check that for you. 🤔"
    return friendly_prefix + raw_response

# --- Small talk / acknowledgment ---
def detect_smalltalk_or_acknowledge(user_input):
    input_lower = user_input.lower()
    if "thank" in input_lower:
        return "You're welcome! Let me know if you have more questions. 😊"
    if "hello" in input_lower or "hi" in input_lower:
        return "Hi there! I'm here to help with anything about Crescent University. What would you like to know?"
    if "not asked" in input_lower:
        return "Alright, take your time! I'm ready whenever you are. 😊"
    return None

# --- Streamlit UI setup ---
st.set_page_config(page_title="Crescent UniBot", page_icon="🎓")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- System prompt for GPT fallback ---
system_prompt = """
You are a helpful, friendly assistant for Crescent University.
Answer in a conversational tone, like you're chatting with a student.
If the user is confused, be patient. If they greet you, respond warmly.
Always encourage them to ask more if needed.
"""

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

        if similarity_score < 0.45:  # fallback threshold reduced for speed & accuracy
            conversation_history.append({"role": "user", "content": prompt})

            with st.spinner("Thinking... 🤔"):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # Faster fallback model
                    messages=[{"role": "system", "content": system_prompt}] + conversation_history,
                    temperature=0.7,
                    max_tokens=500
                )
            gpt_reply = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": gpt_reply})
            final_answer = gpt_reply
        else:
            final_answer = matched_answer
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": final_answer})

        # Limit conversation history size for memory/performance
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]

        human_response = add_human_tone(final_answer)
        st.chat_message("assistant").write(human_response)
        st.session_state.messages.append({"role": "assistant", "content": human_response})
