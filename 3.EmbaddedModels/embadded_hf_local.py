from langchain_huggingface import HuggingFaceEmbeddings

embaddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

text = 'islamabad is the capital of Pakistan'