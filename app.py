import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from load_api_key import load_openai_api_key
from process import process_text


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    st.title("📄Machine Learning Paper Summarizer")
    st.write("Created by Yoonhee Gil")
    st.caption("This page provides detailed summarization of machine learning paper with OpenAI GPT-4o-mini.")
    st.divider()

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        os.environ["OPENAI_API_KEY"] = openai_api_key
    # try:
    #     os.environ["OPENAI_API_KEY"] = load_openai_api_key()
    # except ValueError as e:
    #     os.environ["OPENAI_API_KEY"] = openai_api_key

    pdf = st.file_uploader('Upload your Machine Learning Paper', type='pdf')
    submitted = st.form_submit_button("Submit")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create the knowledge base object
        knowledgeBase = process_text(text)

        query = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

        if query:
            docs = knowledgeBase.similarity_search(query)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
            st.subheader('Summary Results:')
            st.write(response)

if __name__ == '__main__':
    main()