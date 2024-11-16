import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load the environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in a .env file.")

# Initialize the OpenAI LLM to Query a CSV File
def chatwithcsv(df, prompt):
    if openai_api_key:
        llm = OpenAI(api_key=openai_api_key)
        prompt_template = PromptTemplate(input_variables=["data", "prompt"], template="{data}\n\n{prompt}")
        llmchain = LLMChain(llm=llm, prompt=prompt_template)
        result = llmchain.run({"data": df.to_string(), "prompt": prompt})
        print(result)
        return result
    else:
        return "OpenAI API key is missing."


# Set up Streamlit interface
st.set_page_config(layout='wide')
st.title("Dashboard for CSV File powered by LangChain")

# Upload CSV file
input_csv = st.file_uploader("Upload Your CSV File", type=['csv'])

if input_csv is not None:
    # Display the uploaded file in two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("File Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data)

    with col2:
        st.info("Chat with CSV")
        input_text = st.text_area("Enter Your Query")
        if input_text is not None:
            if st.button("Show Results"):
                st.info("Your Query: "+input_text)
                result = chatwithcsv(data, input_text)
                st.success(result)
        