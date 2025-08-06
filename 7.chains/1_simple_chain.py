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

template = PromptTemplate(
    template='Generate 5 intersting facts about {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()
chain = template | model | parser
result = chain.invoke({'topic':'Cricket'})
print(result)
# you can also make grapgh of the chains using the below method
chain.get_graph().print_ascii()