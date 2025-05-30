import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import numpy as np
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from chatbot import load_llm

def calculate_semantic_similarity(reference, candidate, embedding_model):
    ref_embedding = embedding_model.embed_query(reference)
    cand_embedding = embedding_model.embed_query(candidate)
    ref_embedding = np.array(ref_embedding).reshape(1, -1)
    cand_embedding = np.array(cand_embedding).reshape(1, -1)
    return cosine_similarity(ref_embedding, cand_embedding)[0][0]

def evaluate_model():
    st.set_page_config(page_title="PDF Evaluation", page_icon="📊", layout="wide")
    st.title("📄 PDF-Based Evaluation of Diabetes Assistant AI")

    uploaded_pdf = st.file_uploader("Upload a Diabetes PDF for Evaluation", type="pdf")

    if uploaded_pdf is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            st.success(f" Loaded {len(documents)} pages from the uploaded PDF.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            text_chunks = text_splitter.split_documents(documents)
            st.info(f"🔍 Created {len(text_chunks)} text chunks for indexing.")

            embedding_model = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                encode_kwargs={'device': 'cuda'}
            )
            vectorstore = FAISS.from_documents(text_chunks, embedding_model)

            llm = load_llm()
            if llm is None:
                st.error("❌ LLM failed to load. Check your Hugging Face API key.")
                return

            CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer the user's question.
                If you don't know the answer, say that you don't know.
                Context: {context}
                Question: {question}
            """

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])}
            )

            st.subheader("📝 Evaluation Questions")
            questions = st.text_area("Enter your evaluation questions, one per line", height=200)

            if questions.strip() and st.button("Evaluate"):
                questions_list = [q.strip() for q in questions.strip().splitlines() if q.strip()]
                results = []
                total_similarity = 0

                st.info(f"Evaluating {len(questions_list)} questions...")

                for question in questions_list:
                    response = qa_chain.invoke({'query': question})
                    model_answer = response['result'].strip()

                    with st.expander(f"Q: {question}"):
                        reference_answer = st.text_area(f"Reference answer for: {question}", key=question)

                        if reference_answer:
                            score = calculate_semantic_similarity(reference_answer, model_answer, embedding_model)
                            st.markdown(f"**Model Answer:** {model_answer}")
                            st.markdown(f"**Similarity Score:** `{score:.4f}`")
                            results.append(score)
                            total_similarity += score

                if results:
                    avg_similarity = total_similarity / len(results)
                    st.metric("📈 Average Semantic Similarity Score", f"{avg_similarity:.4f}")

        except Exception as e:
            st.error(f"🚨 Error during PDF evaluation: {str(e)}")

if __name__ == "__main__":
    evaluate_model()
