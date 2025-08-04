import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Streamlit UI
st.set_page_config(page_title="Gemini Chatbot", layout="centered")
st.title("🧠 Gemini Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
user_input = st.chat_input("Type your message...")
if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from Gemini
    response = model.invoke(user_input)
    ai_reply = response.content

    # Show AI message
    st.chat_message("assistant").markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
