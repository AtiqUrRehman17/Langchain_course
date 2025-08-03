import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the .env file
load_dotenv()

# Load the Gemini API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Check if the key is loaded properly
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Ask a test question
response = llm.invoke("What is the capital of Pakistan?")
print("Gemini Response:", response.content)
