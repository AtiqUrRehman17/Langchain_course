from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a Joke about {topic}',
    input_variables=['topic']
)
joke_gen_chain = RunnableSequence(prompt1,model,parser)


final_chain = RunnableParallel({
    'Joke':RunnablePassthrough(),
    'word_count':RunnableLambda(lambda x:len(x.split()))
})
final_chain_2 = RunnableSequence(joke_gen_chain,final_chain)
rsult = final_chain_2.invoke({'topic':'AI'})
print(rsult)