from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
import os
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)

# here the pydantic model will be name age and city and the constrain will be that age shoild be > 18 and also it should be interger
class Person(BaseModel):
    name:str = Field(description='name of the person')
    age:int = Field(gt=18,description='age of the person')
    city:str = Field(description='city of the person')
    
parser = PydanticOutputParser(pydantic_object=Person)

template =PromptTemplate(
    template='generate the name,age and city of the fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
chain = template | model | parser

reslut = chain.invoke({'place':'Pakistani'})
print(reslut)

# prompt = template.invoke({"place":'Pakistani'})
# result = model.invoke(prompt)
# fainal_result = parser.parse(result.content)
# print(fainal_result)