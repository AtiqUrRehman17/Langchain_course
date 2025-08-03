from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
model = ChatAnthropic(model= 'clude-3.5',temperature=0.3,max_tokens_to_sample=20)

result = model.invoke('who is the pm of india')
print(result)