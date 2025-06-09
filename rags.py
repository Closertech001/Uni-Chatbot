# --- Imports ---
import streamlit as st
import re
import json
import time
from symspellpy.symspellpy import SymSpell
from sentence_transformers import SentenceTransformer, util
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

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize conversation history
conversation_history = []

# --- Constants for preprocessing ---
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
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        text = suggestions[0].term

    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"\\b{abbr}\\b", full, text, flags=re.IGNORECASE)

    for word, synonym in SYNONYMS.items():
        text = re.sub(rf"\\b{word}\\b", synonym, text, flags=re.IGNORECASE)

    return text

# --- Detect smalltalk or acknowledgements ---
def detect_smalltalk_or_acknowledge(user_input):
    input_lower = user_input.lower()
    if "thank" in input_lower:
        return "You're welcome! Let me know if you have more questions. 😊"
    if "hello" in input_lower or "hi" in input_lower:
        return "Hi there! I'm here to help with anything about Crescent University. What would you like to know?"
    if "not asked" in input_lower:
        return "Alright, take your time! I'm ready whenever you are. 😊"
    return None

# --- Personality detection prompt ---
PERSONALITY_DETECT_PROMPT = """
Analyze the user's message and identify their mood or personality from the following categories:
- friendly
- formal
- curious
- impatient
- humorous
- confused
Respond with only one word representing the detected personality.
User message:
"""

# --- Tone styles mapping ---
TONE_STYLES = {
    "friendly": "You are a helpful and friendly assistant. Use a warm and encouraging tone with emojis where appropriate.",
    "formal": "You are a professional assistant. Use a polite and formal tone.",
    "curious": "You are an engaging assistant, showing curiosity and enthusiasm in your responses.",
    "impatient": "You are a direct and concise assistant, giving answers quickly and efficiently.",
    "humorous": "You are a witty and light-hearted assistant. Use humor and casual language.",
    "confused": "You are a patient and gentle assistant, helping to clarify and explain carefully."
}

# --- Streamlit UI setup ---
st.set_page_config(page_title="Crescent UniBot", page_icon="🎓")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Helper function: detect personality ---
def detect_personality(user_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": PERSONALITY_DETECT_PROMPT + user_text}],
            temperature=0,
            max_tokens=5,
        )
        personality = response.choices[0].message.content.strip().lower()
        if personality not in TONE_STYLES:
            personality = "friendly"
        return personality
    except Exception:
        # Fallback
        return "friendly"

# --- Handle new user input ---
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Smalltalk or quick replies
    friendly_reply = detect_smalltalk_or_acknowledge(prompt)
    if friendly_reply:
        st.chat_message("assistant").write(friendly_reply)
        st.session_state.messages.append({"role": "assistant", "content": friendly_reply})
    else:
        processed_prompt = preprocess_text(prompt)

        # Semantic search
        query_embedding = embedder.encode(processed_prompt, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
        top_hit = hits[0][0]
        matched_answer = qa_data[top_hit['corpus_id']]['answer']
        similarity_score = top_hit['score']

        # Detect personality dynamically from user message
        personality = detect_personality(prompt)
        system_tone_prompt = TONE_STYLES[personality]

        # Prepare conversation for GPT fallback or direct answer
        conversation_history.append({"role": "user", "content": prompt})

        if similarity_score < 0.55:
            # GPT fallback with dynamic tone
            messages = [
                {"role": "system", "content": system_tone_prompt},
                *conversation_history[-10:]
            ]
            # Stream response from GPT
            response_stream = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=True,
            )
            collected_chunks = []
            collected_messages = []
            full_reply = ""
            assistant_message = st.chat_message("assistant")
            partial_response = ""

            for chunk in response_stream:
                chunk_message = chunk['choices'][0]['delta'].get('content', '')
                partial_response += chunk_message
                assistant_message.write(partial_response)
                full_reply = partial_response
            conversation_history.append({"role": "assistant", "content": full_reply})
            final_answer = full_reply
        else:
            # Use matched answer, add human tone prefix based on personality
            tone_prefixes = {
                "friendly": "Of course! 😊 ",
                "formal": "",
                "curious": "Great question! 🤔 ",
                "impatient": "",
                "humorous": "Haha, here you go! 😄 ",
                "confused": "Let me clarify that for you. 🤓 "
            }
            prefix = tone_prefixes.get(personality, "Of course! 😊 ")
            final_answer = prefix + matched_answer
            conversation_history.append({"role": "assistant", "content": final_answer})

        # Limit conversation history size
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

        # Display final response if not streamed already
        if similarity_score >= 0.55:
            st.chat_message("assistant").write(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
