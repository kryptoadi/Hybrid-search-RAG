import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from hybrid_retriever import create_bm25,ask_question

st.title("Hybrid Search RAG")
st.write("Upload PDFs and ask questions about their content.")

uploaded_files=st.file_uploader(
    "Upload Your files",type="pdf",accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process PDFs"):
        documents = []
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(file.name)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata["source"] = file.name

            documents.extend(pdf_docs)
            
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks=splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"]=uploaded_files[0].name
            
        embeddings=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2"
        )
        vectorstore=FAISS.from_documents(chunks,embeddings)
        bm25=create_bm25(chunks)
        st.session_state.chunks = chunks
        st.session_state.vectorstore = vectorstore
        st.session_state.bm25 = bm25
        st.success("PDFs processed successfully!")


if "chunks" in st.session_state:
    query=st.text_input("Ask question about Your Pdfs")
    
    if st.button("Get Answer") and query:
        with st.spinner("Searching documents..."):
            answer, sources = ask_question(
                query,
                st.session_state.chunks,
                st.session_state.bm25,
                st.session_state.vectorstore
            )

        st.subheader("Answer")
        st.write(answer)

        
        st.subheader("Sources")

        for source in sources:
            page = source["page"]

            if isinstance(page, int):
                page += 1

            with st.expander(
                f"{source['source']} — Page {page}"
            ):
                st.write(source["text"])


        
    
        
        
        
        