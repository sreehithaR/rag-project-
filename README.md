 RAG PDF Question Answering System

 Description

This project is a **Retrieval-Augmented Generation (RAG)** based application that allows users to upload PDF documents and ask questions from them.
The system retrieves relevant content using vector search and generates accurate answers using a language model.



Features

*  Upload and process PDF documents
*  Automatic text splitting into chunks
*  Semantic search using FAISS vector database
*  Question answering using LLM (FLAN-T5)
*  Simple UI built with Streamlit



Tech Stack

* Python
* LangChain
* FAISS (Vector Database)
* HuggingFace Transformers
* Streamlit










 How to Run

 1. Clone the repository

git clone https://github.com/sreehithaR/rag-project-.git
cd rag-project-


 2. Install dependencies

pip install -r requirements.txt


3. Run the application


streamlit run app.py




How It Works

1.  PDF is loaded using PyPDFLoader
2.  Text is split into chunks
3.  Embeddings are created using HuggingFace
4.  Stored in FAISS vector database
5.  Relevant chunks are retrieved for a query
6.  LLM generates the final answer



 Demo Video

👉 [Click here to watch demo](PASTE_YOUR_VIDEO_LINK_HERE)



Example Use Case

* Ask questions from research papers
* Extract insights from documents
* Build intelligent document search systems


Future Improvements

* Add support for multiple PDFs
* Improve UI design
* Use advanced LLMs for better accuracy


Author

Sreehitha Reddy

