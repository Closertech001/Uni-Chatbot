import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random
import os

# 🛠️ Set page configuration FIRST
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load data
@st.cache_data
def load_data():
    qa_pairs = []
    try:
        with open("UNIVERSITY DATASET.txt", 'r', encoding='utf-8') as file:
            question, answer = None, None
            for line in file:
                line = line.strip()
                if line.startswith("Q:"):
                    question = line[2:].strip()
                elif line.startswith("A:"):
                    answer = line[2:].strip()
                    if question and answer:
                        qa_pairs.append((question, answer))
                        question, answer = None, None
        df = pd.DataFrame(qa_pairs, columns=["question", "response"])
        return df
    except FileNotFoundError:
        st.error("🚫 Dataset file 'UNIVERSITY DATASET.txt' not found.")
        return pd.DataFrame(columns=["question", "response"])

model = load_model()
dataset = load_data()
question_embeddings = model.encode(dataset['question'].tolist(), convert_to_tensor=True) if not dataset.empty else None

uncertainty_phrases = [
    "I think ", "Maybe this helps: ", "Here's what I found: ",
    "Possibly: ", "It could be: "
]

def find_response(user_input, dataset, question_embeddings, model, threshold=0.6):
    user_input = user_input.strip().lower()

    greetings = [
        "hi", "hello", "hey", "hi there", "greetings", "how are you",
        "how are you doing", "how's it going", "can we talk?",
        "can we have a conversation?", "okay", "i'm fine", "i am fine"
    ]
    if user_input in greetings:
        return random.choice([
            "Hello!", "Hi there!", "Hey!", "Greetings!",
            "I'm doing well, thank you!", "Sure pal", "Okay"
        ]), 1.0

    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    top_score = torch.max(cos_scores).item()
    top_index = torch.argmax(cos_scores).item()

    if top_score < threshold:
        return random.choice([
            "I'm sorry, I don't understand your question.",
            "Can you rephrase your question?"
        ]), top_score

    response = dataset.iloc[top_index]['response']
    if random.random() < 0.2:
        response = random.choice(uncertainty_phrases) + response
    return response, top_score

# Streamlit UI
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask anything about the university. I'm here to help!")
st.divider()

user_input = st.text_input("Enter your question 💬:")

if user_input and not dataset.empty:
    response, top_score = find_response(user_input, dataset, question_embeddings, model)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Chatbot:**", response)
    with col2:
        st.metric(label="Confidence", value=f"{top_score:.2f}")

    # Use a dynamic key for feedback to avoid conflicts for different questions
    feedback_key = f"feedback_{hash(user_input)}"
    feedback = st.radio("Was this answer helpful?", ("", "Yes", "No"), index=0, key=feedback_key)

    # Store feedback status in session state to avoid duplicate logs
    feedback_logged_key = f"feedback_logged_{hash(user_input)}"

    if feedback in ("Yes", "No") and not st.session_state.get(feedback_logged_key, False):
        with open("feedback_log.csv", "a", encoding='utf-8') as f:
            f.write(f"{user_input},{response},{feedback}\n")
        st.session_state[feedback_logged_key] = True
        st.success("✅ Thanks for your feedback!")

    # Disable the radio buttons once feedback is submitted
    if st.session_state.get(feedback_logged_key, False):
        st.radio("Was this answer helpful?", ("Yes", "No"), index=("Yes", "No").index(feedback), key=feedback_key, disabled=True)
