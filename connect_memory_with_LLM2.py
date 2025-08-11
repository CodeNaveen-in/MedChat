import os
# We now import the specific, reliable Groq chat model class - Made by AI Studio as Hugging face one wasn't working.
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# --- Step 1: Setup a Reliable LLM using Groq ---

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

# --- Step 2: Create QA Chain ---

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful medical assistant. Use the pieces of information provided in the context to answer the user's question.
Your answer must be based *only* on the provided context. Do not use any external knowledge. Use that knowledge a bit elaborative to help user get better responses.
If the context does not contain the answer, just say "I'm sorry, the provided information does not contain the answer to this question."

Context: {context}
Question: {question}

Helpful Answer:
"""

def set_custom_prompt(custom_prompt_template):
    """Sets the custom prompt template for the QA chain."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load the existing vector database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create the QA chain with the reliable Groq LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# --- Step 3: Run the Chatbot ---
if __name__ == "__main__":
    print("Your Medical Chatbot (powered by Groq) is ready. Ask a question.")
    user_query = input("Write Query Here: ")
    
    # Run the query and print the response
    response = qa_chain.invoke({'query': user_query})
    
    print("\nRESULT:\n")
    print(response["result"])
    # To see the documents that were retrieved and used as context:
    # print("\n\nSOURCE DOCUMENTS:\n", response["source_documents"])