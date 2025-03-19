import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from textblob import TextBlob
from langchain import hub
import os
import base64

def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Please upload a PDF, TXT, or DOCX file.")

    documents = loader.load()
    return documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def analyze_sentiment(query):
    sentiment = TextBlob(query).sentiment.polarity
    if sentiment > 0.1:
        return "positive"
    elif sentiment < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_prompt(query, sentiment):
    if sentiment == "positive":
        return f"The user is optimistic and curious. Here's the query: {query}"
    elif sentiment == "negative":
        return f"The user seems concerned or upset. Provide a detailed and reassuring response to the query: {query}"
    else:
        return f"The user has a neutral tone. Answer the query directly: {query}"

def ask_rag_chain(query):
    sentiment = analyze_sentiment(query)
    prompt = generate_prompt(query, sentiment)
    return prompt

def main():
    st.set_page_config(layout="wide")
    st.title("AskMyDoc")

    tab1, tab3 = st.tabs(["Upload the File", "Q/A"])

    with tab1:
        st.write("Upload a PDF, Word, or text file to process its content.")
        uploaded_file = st.file_uploader("Upload your file", type=["pdf", "docx", "txt"])

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1]
            st.write(f"File uploaded: {uploaded_file.name} (Extension: {file_extension})")

            file_path = f"uploaded_file.{file_extension}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                docs = load_file(file_path)

                custom_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
                        You are an intelligent AI assistant committed to user privacy and security.

                        - If you detect **explicitly personal or sensitive information** (like SSNs, mobile numbers, passwords), omit it entirely and respond with:
                        *"I cannot provide that information due to privacy and security concerns."*

                        - If the question refers to **general, non-sensitive user data** (e.g., past job details, public work experience), retrieve and provide a clear response.

                        Use the following retrieved context to answer the user’s question as accurately as possible.

                        Context: {context}

                        Question: {question}

                        Provide a response **only if it does not involve private credentials, personally identifiable information (PII), or financial data**. If the context does not contain relevant information, respond with:
                        *"I don't have enough information to answer that."*

                    """
                )

                prompt = custom_prompt

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
                splits = text_splitter.split_documents(docs)
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

                retriever = vectorstore.as_retriever()
                llm = OllamaLLM(model="llama3.2")

                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                )
                st.write("Upload Successful")
            except ValueError as e:
                st.error(f"Error processing file: {e}")

    with tab3:
        if uploaded_file is not None:
            st.subheader("Q/A")
            question = st.text_input('Please enter your question:')
            if question:
                st.write(rag_chain.invoke(ask_rag_chain(question)))

if __name__ == "__main__":
    main()