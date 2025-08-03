from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='anything model name',temperature=0.5,max_completion_tokens=24)

result = model.invoke('hi how are you')
print(result)