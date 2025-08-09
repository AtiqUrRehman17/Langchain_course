from langchain.text_splitter import RecursiveCharacterTextSplitter

text = '''Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks 
typically requiring human intelligence. These tasks include understanding language, recognizing 
patterns, solving problems, learning from experience, and making decisions. AI aims to create 
machines that can simulate aspects of human thinking and behavior. Unlike traditional software 
that follows fixed instructions, AI systems can improve themselves over time through learning 
and adaptation. 
AI is not a single technology but a field that combines various disciplines, including computer 
science, mathematics, neuroscience, linguistics, and more. From virtual assistants like Siri and 
Alexa to complex systems like autonomous vehicles and fraud detection algorithms, AI is 
transforming the way we live and work'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separators=''
)

result = splitter.split_text(text)
print(len(result))
print(result)