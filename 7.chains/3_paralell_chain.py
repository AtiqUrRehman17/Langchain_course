from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# Initialize the model using the API key explicitly
model1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = api_key)


llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation'
)


model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate notes form the foloowing text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 Question answering quize from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Mearge the provided notes and quize into a single dacument \n notes -> {notes} and quize {quize}',
    input_variables=['notes','quize']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quize':prompt2 |model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = '''Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for classification and regression tasks, though it's mostly known for classification.

At its core, SVM aims to find the best separating hyperplane that divides data points of different classes. This hyperplane maximizes the margin—the distance between the hyperplane and the nearest data points from each class, known as support vectors.

Key Concepts:
Hyperplane: A decision boundary that separates different classes.

Support Vectors: Data points closest to the hyperplane; most critical in defining the margin.

Margin: The distance between the support vectors and the hyperplane; SVM tries to maximize it.

Kernel Trick: For non-linearly separable data, SVM uses kernel functions (e.g., polynomial, RBF) to transform data into higher dimensions where a linear separator is possible.

Advantages:
Effective in high-dimensional spaces.

Works well for clear margin of separation.

Memory efficient (uses only support vectors for decision function).

Disadvantages:
Can be less effective with noisy or overlapping classes.

Computationally intensive for large datasets.

Common Use Cases:
Image classification

Spam detection

Bioinformatics (e.g., cancer classification)

Face detection
'''
result = chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii()