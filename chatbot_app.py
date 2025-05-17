
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load data
@st.cache_data
def load_data():
    qa_pairs = []
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

model = load_model()
dataset = load_data()
question_embeddings = model.encode(dataset['question'].tolist(), convert_to_tensor=True)

uncertainty_phrases = [
    "I think ", "Maybe this helps: ", "Here's what I found: ",
    "Possibly: ", "It could be: "
]

def find_response(user_input, dataset, question_embeddings, model, threshold=0.6):
    greetings = [
        "hi", "hello", "hey", "hi there", "greetings", "how are you",
        "how are you doing", "how's it going", "can we talk?",
        "can we have a conversation?", "okay", "i'm fine", "i am fine"
    ]
    if user_input.lower() in greetings:
        return random.choice([
            "Hello!", "Hi there!", "Hey!", "Greetings!",
            "I'm doing well, thank you!", "Sure pal", "Okay"
        ])

    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    top_score = torch.max(cos_scores).item()
    top_index = torch.argmax(cos_scores).item()

    if top_score < threshold:
        return random.choice([
            "I'm sorry, I don't understand your question.",
            "Can you rephrase your question?"
        ])

    response = dataset.iloc[top_index]['response']
    if random.random() < 0.2:
        response = random.choice(uncertainty_phrases) + response
    return response

# Streamlit UI
st.title("🎓 University Q&A Chatbot")

user_input = st.text_input("Ask a question:")
if user_input:
    response = find_response(user_input, dataset, question_embeddings, model)
    st.write("**Chatbot:**", response)
