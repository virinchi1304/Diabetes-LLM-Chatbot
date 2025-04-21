import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import torch
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={'device': device}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def load_llm():
    HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
    if not HF_TOKEN:
        st.error("🚨 Hugging Face API key not found in .env")
        return None
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        token=HF_TOKEN,
        task="text-generation",
        temperature=0.5,
        max_length=512
    )

def reset_chat():
    if "messages" in st.session_state and st.session_state.messages:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({"timestamp": timestamp, "messages": st.session_state.messages})
    st.session_state.messages = []

def restore_chat(index):
    if 0 <= index < len(st.session_state.chat_history):
        st.session_state.messages = st.session_state.chat_history[index]["messages"]

def main():
    st.set_page_config(page_title="🧠 Diabetes Assistant", page_icon="🩺", layout="wide")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("🧠 Diabetes Assistant")
    col1, col2 = st.columns([3, 1])
    with col2:
        st.button("🆕 New Chat", on_click=reset_chat)

    with st.sidebar:
        st.title("📜 Chat History")
        if st.session_state.chat_history:
            for idx, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"Chat {idx + 1} ({chat['timestamp']})"):
                    preview = " | ".join([msg['content'][:50] for msg in chat['messages'][:3]]) + "..."
                    st.text(preview)
                    if st.button(f"🔄 Restore Chat {idx + 1}", key=f"restore_{idx}"):
                        restore_chat(idx)
                        st.rerun()
        else:
            st.info("No chat history yet. Start chatting!")

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer the user's question.
            If you don't know the answer, say that you don't know.
            Context: {context}
            Question: {question}
        """

        try:
            vectorstore = get_vectorstore()
            llm = load_llm()
            if not vectorstore or not llm:
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"].replace('\n', ' ').strip()

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")

if __name__ == "__main__":
    main()
