from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embadding =OpenAIEmbeddings(model='model name',dimensions=32)

result = embadding.embed_query('what is the pm of pakistan')
print(str(result))  # first we will convert the vector to string then print