import os
import streamlit as st
from pypdf import PdfReader
from bs4 import BeautifulSoup
from urllib.request import urlopen
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def get_text_from_document(file):
    pdf = PdfReader(file)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

def get_text_from_url(url):
    page = urlopen(url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text

def get_faqs(content):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": 'system',
                'content': """Your role is to help users extract content from documents contents like terms of services they provide. Once you have the content, you will thoroughly read through it and generate questions that could be considered for a FAQ section for each content segment. Additionally, you will create concise answers for these questions, referencing the specific section or topic from which the answer was derived. carefully analyzing documents to identify key points that could form the basis of frequently asked questions.
Ensure accuracy in content extraction and interpretation. Use markdown format to respond. Mention where you get the answer from at last of the respective answer. Generate as mch question and answer pairs for the content based upon the information present in the content.
Here is the following format to response:
## <Heading or section name>
### Q: <question>
A: <answer>
*source: <source where you get that info>*"""
            },
            {
                'role': 'user',
                'content': f"Here is the content I would like to generate FAQs for: {content}\nGenerate FAQs for each topic."
            }
        ],
            model="gpt-4o",
            temperature=0.7,
            top_p=0.9,
            stream=False,
        )
    return chat_completion

def main():
    st.title("Contents to FAQ")

    st.write("Enter the URL or upload document")
    url = st.text_input("URL")
    file = st.file_uploader("Upload File")
    process = st.button("Process")

    if process and (url or file):
        if url:
            text = get_text_from_url(url)
        else:
            text = get_text_from_document(file)

        try:
            message_placeholder = st.empty()
            response = get_faqs(text)
            st.write(f"## Usage detailes:\n- completion token (output): {response.usage.completion_tokens}\n- prompt token (input): {response.usage.prompt_tokens}\n- total tokens: {response.usage.total_tokens}")
            message_placeholder.markdown(response.choices[0].message.content)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()