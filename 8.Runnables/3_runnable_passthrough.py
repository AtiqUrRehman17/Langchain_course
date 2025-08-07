from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel,RunnablePassthrough
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)

prompt1 = PromptTemplate(
    template='Generate a Joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the Joke {topic}',
    input_variables=['topic']
)



parser = StrOutputParser()

joke_generator = RunnableSequence(prompt1,model,parser)

paralle_chain = RunnableParallel({
    'Joke':RunnablePassthrough(),
    'Explanation':RunnableSequence(prompt2,model,parser)
})
final_chain = RunnableSequence(joke_generator,paralle_chain)

result = final_chain.invoke({"topic":'AI'})
print(result)