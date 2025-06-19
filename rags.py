# crescent_chatbot.py

import streamlit as st
import re
import json
import time
import random
import torch
import openai
import os
from symspellpy.symspellpy import SymSpell
import pkg_resources
from sentence_transformers import SentenceTransformer, util

# --- Setup Page ---
st.set_page_config(page_title="Crescent University Chatbot", layout="centered", page_icon="🎓")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University!")

# --- Constants ---
USER_PROFILE_PATH = "user_profile.json"
MEMORY_DIR = "memories"
os.makedirs(MEMORY_DIR, exist_ok=True)

# --- API Key ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Caching Resources ---
@st.cache_resource()
def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

@st.cache_resource()
def load_model_and_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [q["question"] for q in data]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return model, data, embeddings

sym_spell = load_symspell()
model, qa_data, qa_embeddings = load_model_and_data()

# --- Preprocessing Maps ---
ABBREVIATIONS = {"u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "plz": "please", "pls": "please", "tmrw": "tomorrow",
    "wat": "what", "yr": "year", "sem": "semester", "admsn": "admission", "clg": "college", "sch": "school",
    "uni": "university", "cresnt": "crescent", "msg": "message", "idk": "i don't know", "imo": "in my opinion",
    "asap": "as soon as possible", "dept": "department", "reg": "registration", "fee": "fees",
    "pg": "postgraduate", "app": "application", "req": "requirement", "nd": "national diploma",
    "1st": "first", "2nd": "second", "nxt": "next", "prev": "previous", "exp": "experience", "b4": "before"}

SYNONYMS = {"lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff",
    "hod": "head of department", "course": "subject", "class": "course", "unit": "credit",
    "hostel": "accommodation", "lodging": "accommodation", "fees": "tuition",
    "school fees": "tuition", "acceptance fee": "admission fee"}

# --- Normalization ---
def normalize_text(text):
    text = text.lower()
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"\\b{abbr}\\b", full, text)
    for word, synonym in SYNONYMS.items():
        text = re.sub(rf"\\b{word}\\b", synonym, text)
    return text

def correct_text(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# --- QA Search ---
def search_answer(query, threshold=0.65):
    normalized = normalize_text(query)
    corrected = correct_text(normalized)
    user_embedding = model.encode(corrected, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, qa_embeddings)[0]
    best_idx = torch.argmax(similarities).item()
    best_score = similarities[best_idx].item()
    if best_score > threshold:
        return qa_data[best_idx]["answer"]
    return None

# --- GPT Fallback ---
def get_gpt_answer(user_prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful, friendly university assistant."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Sorry, I'm unable to fetch that right now."

# --- User Profile ---
def load_user_profile():
    if os.path.exists(USER_PROFILE_PATH):
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"name": None, "department": None, "personality": None}

def save_user_profile(profile):
    with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def update_user_profile_from_input(text, profile):
    text = text.lower()
    name_match = re.search(r"my name is (\w+)", text)
    if name_match:
        profile["name"] = name_match.group(1).capitalize()
    dept_match = re.search(r"(i'?m|i am|i study in|i am in) (department of|dept of)? ?([\w\s]+)", text)
    if dept_match:
        dept = dept_match.group(3).strip().title()
        profile["department"] = dept if len(dept.split()) <= 4 else profile["department"]
    if any(word in text for word in ["i like jokes", "i'm serious", "i like facts", "i love chatting"]):
        if "joke" in text:
            profile["personality"] = "playful"
        elif "serious" in text or "facts" in text:
            profile["personality"] = "serious"
        elif "chat" in text:
            profile["personality"] = "friendly"
    return profile

# --- Personalization ---
def personalize_response(response, profile):
    personality = profile.get("personality", "neutral")
    name = profile.get("name", "")
    if personality == "playful":
        return f"😄 Gotcha, {name or 'friend'}! Here's something fun:\n\n" + response
    elif personality == "serious":
        return f"Certainly {name or 'student'}, here is the information you requested:\n\n" + response
    elif personality == "friendly":
        return f"Hey {name or 'there'}! 😊 Here's what I found:\n\n" + response
    else:
        return f"Of course! 😊 {response}"

# --- Long-Term Memory (per user) ---
def get_user_memory_path(user_id):
    return os.path.join(MEMORY_DIR, f"user_memory_{user_id.lower()}.json")

def load_user_memory(user_id):
    path = get_user_memory_path(user_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_memory(user_id, query, response):
    path = get_user_memory_path(user_id)
    memory = load_user_memory(user_id)
    memory.append({"query": query, "response": response})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def recall_user_memory(user_id, query, threshold=0.7):
    memory = load_user_memory(user_id)
    if not memory:
        return None
    questions = [m["query"] for m in memory]
    embeddings = model.encode(questions, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    sim_scores = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = torch.argmax(sim_scores).item()
    best_score = sim_scores[best_idx].item()
    if best_score > threshold:
        return memory[best_idx]["response"]
    return None

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me anything about Crescent University 🏫"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    user_profile = load_user_profile()
    user_profile = update_user_profile_from_input(prompt, user_profile)
    save_user_profile(user_profile)
    user_id = user_profile.get("name", "anonymous")

    answer = search_answer(prompt)
    if not answer:
        answer = recall_user_memory(user_id, prompt)
    if not answer:
        answer = get_gpt_answer(prompt)

    response = personalize_response(answer, user_profile)
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_user_memory(user_id, prompt, response)

# --- Sidebar: Profile ---
with st.sidebar:
    st.markdown("👤 **User Profile**")
    user_profile = load_user_profile()  # 🔥 Add this line to avoid NameError
    st.write(user_profile)
