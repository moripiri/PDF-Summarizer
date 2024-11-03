from PyPDF2 import PdfReader
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from typing import Optional


class ResearchPaperExtraction(BaseModel):
    Title: str = Field(description="Title of the paper")
    Authors: list[str,...] = Field(description="Authors of the paper in form of Name(Institution)")
    Publication_date: int = Field(description="Publication Year and Month of the paper in form of Year/Month (ex. 2025/12)")
    Official_code_link: Optional[str] = Field(description="Link of the official code of the paper, if available")
    Abstract: str = Field(description="Summarize the abstract into 1 sentence")
    Introduction: str = Field(description="Summarize the abstract into 1~5 bulleted list.")
    Backgrounds: str = Field(description="Summarize the backgrounds into 1~5 bulleted list.")
    Related_Works: str = Field(description="Summarize the related works into 1~5 bulleted list.")
    Methodology_or_Approach: str = Field(description="Summarize the methodology into 1~5 bulleted list.")
    Experiments: str = Field(description="Summarize the experiments into 1~5 bulleted list.")
    Results: str = Field(description="Summarize the results into 1~5 bulleted list.")
    Conclusion:str = Field(description="Summarize the conclusion into 1~5 bulleted list.")

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    st.title("📄Machine Learning Paper Summarizer")
    st.write("Created by Yoonhee Gil")
    st.caption("This page provides detailed summarization of machine learning paper with OpenAI GPT-4o-mini.")
    st.caption(":arrow_upper_left: Submit your OpenAI API key")
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

    pdf = st.file_uploader('Upload your Machine Learning Paper', type='pdf')
    run = st.button("Submit", type="secondary")
    st.divider()

    if run:
        if (pdf is not None) and (openai_api_key != ""):
            pdf_reader = PdfReader(pdf)
            # Text variable will store the pdf text
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            llm = ChatOpenAI(model="gpt-4o-mini",
                             temperature=0.0,
                             openai_api_key=openai_api_key)
            structured_llm = llm.with_structured_output(ResearchPaperExtraction)
            with get_openai_callback() as cost:
                response = structured_llm.invoke(f"Summarize this machine learning paper by given output structure. Put important words in bold type. Paper: {text}")

                print(response)

                st.header(response.Title)
                authors = ""
                for name in response.Authors:
                    authors += f"**{name}** " + ", "
                authors = authors[:-2]
                st.caption(authors)

                st.caption(f"Publication Date: {response.Publication_date}")
                if response.Official_code_link is not None:
                    st.caption(f"Code: [link]({response.Official_code_link})")

                st.write(f"**Summary**: {response.Abstract}")
                st.divider()
                st.subheader("Introduction")
                st.write(response.Introduction)
                st.divider()
                st.subheader("Backgrounds")
                st.write(response.Backgrounds)
                st.divider()
                st.subheader("Related Works")
                st.write(response.Related_Works)
                st.divider()
                st.subheader("Methodology")
                st.write(response.Methodology_or_Approach)
                st.divider()
                st.subheader("Experiments")
                st.write(response.Experiments)
                st.divider()
                st.subheader("Results")
                st.write(response.Results)
                st.divider()
                st.subheader("Conclusion")
                st.write(response.Conclusion)
                st.divider()
                print(cost)

        elif (pdf is None) and (openai_api_key != ""):
            st.write("PDF is not submitted.")
        elif (pdf is not None) and (openai_api_key == ""):
            st.write("OpenAI API key is not submitted.")
        elif (pdf is None) and (openai_api_key == ""):
            st.write("PDF and OpenAI API key are not submitted.")
    else:
        st.write("Waiting for you...")


if __name__ == '__main__':
    main()