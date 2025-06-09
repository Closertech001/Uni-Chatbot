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

# --- Constants ---
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
    "CSC": "department of Computer Science", "Mass comm": "department of Mass Communication",
    "law": "department of law", "Acc": "department of Accounting"
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

SMALL_TALK = [
    (re.compile(r"how are you", re.IGNORECASE), ["I'm great, thanks for asking! How can I assist you?", "Doing well! Let me know if you have any university-related questions."]),
    (re.compile(r"who (are|r) you", re.IGNORECASE), ["I'm your Crescent University assistant bot! Ask me anything."]),
    (re.compile(r"(what can you do|help me)", re.IGNORECASE), ["I can answer questions about departments, courses, fees, admissions, and more at Crescent University."])
]

ABUSE_WORDS = ["fuck", "shit", "bitch", "nigga", "dumb", "sex"]
ABUSE_PATTERN = re.compile(r'\\b(' + '|'.join(map(re.escape, ABUSE_WORDS)) + r')\\b', re.IGNORECASE)

DEPARTMENT_NAMES = [d.lower() for d in [
    "Computer Science", "Mass Communication", "Law", "Microbiology",
    "Accounting", "Political Science", "Business Administration", "Business Admin"
]]

# --- OpenAI API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Cache Loaders ---
@st.cache_resource
def init_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym_spell

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_dataset():
    with open("qa_dataset.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_all_data():
    embed_model = load_embedding_model()
    sym_spell = init_symspell()
    dataset = load_dataset()
    questions = [item["question"] for item in dataset]
    q_embeds = embed_model.encode(questions, convert_to_tensor=True, normalize_embeddings=True)
    return embed_model, sym_spell, dataset, q_embeds

# --- Normalization ---
def normalize_text(text, sym_spell):
    text = text.lower()
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected = suggestions[0].term if suggestions else text
    for abbr, full in ABBREVIATIONS.items():
        corrected = re.sub(rf'\\b{re.escape(abbr)}\\b', full, corrected)
    for syn, rep in SYNONYMS.items():
        corrected = re.sub(rf'\\b{re.escape(syn)}\\b', rep, corrected)
    return corrected

# --- Greetings / Farewell ---
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

def get_random_greeting_response():
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! Feel free to ask me anything about Crescent University.",
        "Greetings! How may I be of service?",
        "Hello! Ready to help you with any questions."
    ])

def is_farewell(text):
    return text.lower().strip() in ["bye", "goodbye", "see you", "later", "farewell", "cya", "peace", "exit"]

def get_random_farewell_response():
    return random.choice([
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back anytime.",
        "Bye! Take care!",
        "Farewell! Let me know if you need anything else.",
        "Peace out! Hope to chat again soon."
    ])

# --- Memory ---
def update_chat_memory(norm_input, memory):
    for dept in DEPARTMENT_NAMES:
        if re.search(rf"\\b{re.escape(dept)}\\b", norm_input):
            memory["department"] = dept.title()
            break
    dep_match = re.search(r"department of ([a-zA-Z &]+)", norm_input)
    if dep_match:
        memory["department"] = dep_match.group(1).title()
    lvl_match = re.search(r"(100|200|300|400|500)\\s*level", norm_input)
    if lvl_match:
        memory["level"] = lvl_match.group(1)
    return memory

def resolve_follow_up(raw_input, memory):
    text = raw_input.strip().lower()
    if m := re.match(r"what about (\\d{3}) level", text):
        if memory.get("department"):
            return f"What are the {m.group(1)} level courses in {memory['department']}?"
    if text.startswith("what about") and memory.get("level") and memory.get("department"):
        return f"What are the {memory['level']} level courses in {memory['department']}?"
    if m2 := re.match(r"do they also .* in ([a-zA-Z &]+)\\?", text):
        if memory.get("topic"):
            return f"Do they also offer {memory['topic']} in {m2.group(1).title()}?"
    if m3 := re.match(r"how about .* for ([a-zA-Z &]+)\\?", text):
        if memory.get("topic"):
            return f"Do they also offer {memory['topic']} in {m3.group(1).title()}?"
    return raw_input

# --- Retrieval / GPT Fallback ---
def build_contextual_prompt(messages, memory):
    mem = f"- Department: {memory.get('department') or 'unspecified'}\\n- Level: {memory.get('level') or 'unspecified'}\\n- Topic: {memory.get('topic') or 'unspecified'}"
    return [{"role": "system", "content": "You are a helpful assistant for Crescent University.\\n" + mem}] + messages[-12:]

def retrieve_or_gpt(user_input, dataset, q_embeds, embed_model, messages, memory):
    for item in dataset:
        if user_input.strip().lower() == item["question"].strip().lower():
            return item["answer"], 1.0
    user_embed = embed_model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.pytorch_cos_sim(user_embed, q_embeds)[0]
    best_idx = scores.argmax().item()
    top_score = float(scores[best_idx])
    if top_score >= 0.60:
        return dataset[best_idx]["answer"], top_score
    if openai.api_key:
        try:
            prompt = build_contextual_prompt(messages, memory)
            prompt.append({"role": "user", "content": user_input})
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=prompt,
                temperature=0.4,
            )
            return gpt_response.choices[0].message.content, top_score
        except:
            return "Sorry, I couldn’t fetch a proper response right now.", top_score
    return "I’m not sure what you mean. Could you try rephrasing?", top_score

# --- Logging ---
def log_to_long_term_memory(user_input, assistant_response):
    os.makedirs("logs", exist_ok=True)
    entry = {"user": user_input, "assistant": assistant_response, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open("logs/chat_history_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

# --- Main App ---
def main():
    st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
    st.title("🎓 Crescent University Chatbot")
    st.markdown("Ask me anything about your department, courses, or the university.")

    if "embed_model" not in st.session_state:
        embed_model, sym_spell, dataset, q_embeds = load_all_data()
        st.session_state.embed_model = embed_model
        st.session_state.sym_spell = sym_spell
        st.session_state.dataset = dataset
        st.session_state.q_embeds = q_embeds
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Crescent University assistant. Ask me anything!"}]
        st.session_state.memory = {"department": None, "topic": None, "level": None}

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your question here...")
    if user_input:
        norm_input = normalize_text(user_input, st.session_state.sym_spell)
        if ABUSE_PATTERN.search(norm_input):
            response = "Sorry, I can’t help with that."
        elif is_greeting(norm_input):
            response = get_random_greeting_response()
        elif is_farewell(norm_input):
            response = get_random_farewell_response()
        else:
            for pattern, replies in SMALL_TALK:
                if pattern.search(norm_input):
                    response = random.choice(replies)
                    break
            else:
                st.session_state.memory = update_chat_memory(norm_input, st.session_state.memory)
                resolved_input = resolve_follow_up(user_input, st.session_state.memory)
                response, _ = retrieve_or_gpt(resolved_input, st.session_state.dataset, st.session_state.q_embeds, st.session_state.embed_model, st.session_state.messages, st.session_state.memory)
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Typing..._")
            time.sleep(1.2)
            placeholder.markdown(response)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        log_to_long_term_memory(user_input, response)

if __name__ == "__main__":
    main()
