import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Check if API key is loaded
if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

# Initialize Gemini model
model1 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

# Initialize HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation'
)
model2 = ChatHuggingFace(llm=llm)

# Define prompts
prompt1 = PromptTemplate(
    template='Generate notes from the following text:\n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 Question answering quiz from the following text:\n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document.\nNotes -> {notes}\nQuiz -> {quize}',
    input_variables=['notes', 'quize']
)

# Output parser
parser = StrOutputParser()

# LangChain runnable setup
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quize': prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser

# -------- Streamlit UI --------
st.set_page_config(page_title="Notes & Quiz Generator", layout="wide")
st.title("📘 AI-Powered Notes & Quiz Generator")
st.markdown("Enter your text below to generate **notes**, **quiz**, and a **merged summary document**.")

# Text input
text_input = st.text_area(
    "✍️ Enter your educational text here:",
    height=300,
    value="""Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for classification and regression tasks, though it's mostly known for classification.

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

Face detection"""
)

# Submit button
if st.button("🚀 Generate Notes & Quiz"):
    if not text_input.strip():
        st.warning("Please enter some text to process.")
    else:
        with st.spinner("Generating with Gemini & HuggingFace..."):
            try:
                # Step 1: Get notes & quiz
                intermediate = parallel_chain.invoke({'text': text_input})

                # Step 2: Merge them
                merged = merge_chain.invoke({
                    'notes': intermediate['notes'],
                    'quize': intermediate['quize']
                })

                # Display outputs
                st.success("✅ Generation Complete!")

                st.subheader("📝 Notes")
                st.markdown(intermediate['notes'])

                st.subheader("❓ Quiz")
                st.markdown(intermediate['quize'])

                st.subheader("📄 Merged Document")
                st.markdown(merged)

            except Exception as e:
                st.error(f"❌ Something went wrong:\n{e}")
                
