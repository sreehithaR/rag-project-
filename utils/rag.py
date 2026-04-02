from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
import streamlit as st


@st.cache_resource
def load_embeddings():
    """Load embedding model once and cache it."""

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



@st.cache_resource
def load_llm():
    """Load the LLM pipeline once and cache it."""
    pipe = pipeline(
        "text2text-generation",

        model="google/flan-t5-base",

        max_length=256,
    )
    return HuggingFacePipeline(pipeline=pipe)


def process_docs(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = load_embeddings()
    db = FAISS.from_documents(texts, embeddings)

    return db



from langchain.prompts import PromptTemplate

def ask_question(db, query):
    llm = load_llm()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the question clearly and concisely.
Do not repeat sentences.
If the answer is not present in the context, say "Not found in document".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}

    )

    return qa.run(query)
