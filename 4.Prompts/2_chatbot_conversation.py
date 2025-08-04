from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from environment
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)

# Simple loop to interact
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = model.invoke(user_input)
    print("AI:", response.content)
