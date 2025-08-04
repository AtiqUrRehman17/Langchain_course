from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Safety check
if not api_key:
    st.error("GEMINI_API_KEY is not set. Please check your .env file.")
    st.stop()

# Streamlit UI
st.header('Research Paper Explainer Tool')

paper_input = st.selectbox(
    "Select Research Paper Name", 
    [
        "Attention Is All You Need", 
        "BERT: Pre-training of Deep Bidirectional Transformers", 
        "GPT-3: Language Models are Few-Shot Learners", 
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length", 
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Inline Prompt Template
template = PromptTemplate.from_template(
    "Explain the research paper titled '{paper}' in a {style} style. The explanation should be {length}."
)

# Run LLM on button click
if st.button('Summarize'):
    final_prompt = template.format(
        paper=paper_input,
        style=style_input,
        length=length_input
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key
    )

    with st.spinner("Generating explanation..."):
        try:
            response = llm.invoke(final_prompt)
            st.success("Explanation generated successfully!")
            st.markdown(response.content)
        except Exception as e:
            st.error(f"Error generating explanation: {e}")
