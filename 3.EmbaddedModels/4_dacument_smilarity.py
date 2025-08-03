from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# No 'dimensions' param unless explicitly supported
embeddings = GoogleGenerativeAIEmbeddings(model='gemini-2.0-flash')

documents = [
    'Babar Azam is the captain of Pakistan’s white-ball teams and one of the most consistent batters in world cricket.',
    'Shaheen Shah Afridi is known for his deadly left-arm pace and ability to swing the new ball.',
    'Iftikhar Ahmed is a middle-order batter and occasional off-spinner who brings stability and power-hitting to the T20 side.',
    'Abrar Ahmed, a mystery spinner, has made an impact in Test cricket with his variations and control.',
    'Fakhar Zaman is an explosive left-handed batter known for his big hundreds in limited-overs formats.',
    'Imam-ul-Haq is a reliable left-handed opener who plays a key role in Pakistan’s ODI lineup.',
    'Haris Rauf is a fiery fast bowler famous for his pace and wicket-taking ability in T20 cricket.',
    'Mohammad Rizwan plays as a wicketkeeper-batter and has formed a strong T20 opening partnership with Babar.',
    'Shaheen Shah Afridi is known for his deadly left-arm pace and ability to swing the new ball.'
]

query = 'tell me about Babar Azam'

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

print(cosine_similarity([query_embedding], doc_embeddings))
