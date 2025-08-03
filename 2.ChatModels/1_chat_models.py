from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model='gpt-4')
result = model.invoke('who is the PM of Pakistan,',temprature=0.4,max_completion_tokens=10)
print(result)

# here in both example i don't have the openai api key