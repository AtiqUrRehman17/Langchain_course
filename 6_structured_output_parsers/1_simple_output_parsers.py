from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)

# Prompt templates
template1 = PromptTemplate(
    template='Write me a detailed report on the {topic}.',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write me a five-line summary of the following text:\n{text}',
    input_variables=['text']
)

# Run prompts
prompt1 = template1.invoke({'topic': 'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})
result1 = model.invoke(prompt2)

# Output
print(result1.content)
