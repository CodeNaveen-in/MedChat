import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    """Sets the custom prompt template for the QA chain."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    """Loads the Groq LLM (Llama 3). This is fast and reliable."""
    # Check for the API key
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in .env file.")
        
    # We use ChatGroq, specifying the model we want to use.
    # 'llama3-8b-8192' is an excellent and fast choice.
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=1 # Increased temperature for more factual, more creative answers
    )
    return llm

def main():
    st.title("Ask Chatbot")
    if("messages" not in st.session_state):
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    prompt=st.chat_input("Pass your prompt here")
    if (prompt):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role":"user", "content":prompt})
        CUSTOM_PROMPT_TEMPLATE = """
            You are a helpful medical assistant. Use the pieces of information provided in the context to answer the user's question.
            Your answer must be based *only* on the provided context. Do not use any external knowledge. Use that knowledge a bit elaborative to help user get better responses.
            If the context does not contain the answer, just say "I'm sorry, the provided information does not contain the answer to this question."

            Context: {context}
            Question: {question}

            Helpful Answer:
            """
        
        try:
            vectorstore=get_vectorstore()
            if (vectorstore is None):
                st.error("The Vector Store is Not Loading properly..")
            qa_chain = RetrievalQA.from_chain_type(
                llm = ChatGroq(
                model_name="llama3-8b-8192",
                temperature=1, 
                groq_api_key = os.environ["GROQ_API_KEY"]), # Increased temperature for more factual, more creative answers
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response=qa_chain.invoke({"query":prompt})
            result=response["result"]
            source_documents = response["source_documents"]
            result_show = result + "\nSource Doc: \n" + str(source_documents)
            st.chat_message("assistant").markdown(result_show)
            st.session_state.messages.append({"role":"assistant", "content":result_show})
        
        except Exception as e:
            st.error(f"Error : {str(e)}")

if (__name__ == "__main__"):
    main()