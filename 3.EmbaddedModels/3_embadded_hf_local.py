from langchain_huggingface import HuggingFaceEmbeddings

embaddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

dacuments = [
    'islamabad is the capital of pakistan',
    'delhi is the capital of india',
    'paris is the capital of france'
]

vector = embaddings.embed_documents(dacuments)
print(str(vector))