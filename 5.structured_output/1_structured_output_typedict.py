from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated
from dotenv import load_dotenv

import os

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from environment
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)

# define schema
class review(TypedDict):
    summary:Annotated[str,'A brief summary of the input']
    sentitment:Annotated[str,'return sentiment']
    
structured_model = model.with_structured_output(review)
result = structured_model.invoke('''This phone offers great performance with smooth multitasking and fast app launches.
The camera quality is impressive, especially in daylight and portrait modes.
Battery life easily lasts a full day, and fast charging is a big bonus.''')
print(result)