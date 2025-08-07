from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Writge a detail report on the  {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summerize the following text {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1,model,parser)

brach_chain = RunnableBranch(
    (lambda x:len(x.split())>500 , RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

fainal_chain = RunnableSequence(report_gen_chain,brach_chain)
print(fainal_chain.invoke({'topic':'AI'}))