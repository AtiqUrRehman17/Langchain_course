from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic':'black hole'})
print(result)