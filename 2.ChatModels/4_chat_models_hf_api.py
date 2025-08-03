from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get token from environment
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN is missing. Please set it in the .env file.")

# Set the token so huggingface_hub can use it
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

# Use a chat-supported endpoint (Zephyr is good and lightweight)
llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print(result.content)
