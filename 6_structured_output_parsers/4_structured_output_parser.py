from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model using the API key explicitly
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)

# here first define schema with response schema
schema = [
    ResponseSchema(name='Fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='Fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='Fact_3',description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

tempate = PromptTemplate(
    template='give me 3 facts about {topic}\n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
prompt = tempate.invoke({'topic':'black hole'})
result = model.invoke(prompt)
final_result =parser.parse(result.content)
print(final_result)