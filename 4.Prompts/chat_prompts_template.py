from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate

chat_tempalte = ChatPromptTemplate([
    ('system','You are Helpfull {domain} Expert'),
    ('Human','Tell me about what is {topic}')
])

prompt = chat_tempalte.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)