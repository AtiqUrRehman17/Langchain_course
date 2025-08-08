from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='write the summary of the fowwing - \n {poem}',
    input_variables=['peom']
)

loader = TextLoader('text.txt',encoding='utf-8')

docs = loader.load()

# print(type(docs))
# print(len(docs))

chain = prompt1 | model | parser
result = chain.invoke({'poem':docs[0].page_content})
print(result)