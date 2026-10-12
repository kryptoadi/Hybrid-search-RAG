from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)
loader=PyPDFLoader("data/support-policy.pdf")
documents=loader.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,chunk_overlap=150
)

chunks=splitter.split_documents(documents)


vectorstore=FAISS.from_documents(
    documents=chunks,embedding=embeddings
    
)
vectorstore.save_local("vectorstore")

print("FAISS index created successfully!")