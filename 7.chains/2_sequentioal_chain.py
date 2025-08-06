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

prompt1 = PromptTemplate(
    template='Generate a detail report on the  {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary form the text {text}\n',
    input_variables=['text']
)

parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Unemployement in Pakistan'})
print(result)

chain.get_graph().print_ascii()