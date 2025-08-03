from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embadding =OpenAIEmbeddings(model='model name',dimensions=32)
dacuments = [
    'islamabad is the capital of pakistan',
    'delhi is the capital of india',
    'paris is the capital of france'
]
result = embadding.embed_documents(dacuments)
print(str(result))  # first we will convert the vector to string then print