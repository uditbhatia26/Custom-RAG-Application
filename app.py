import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "CUSTOM RAG QNA"
os.environ["LANGCHAIN_TRACKING_V2"] = 'true'


# Creating the embeddings
embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')


# Creating the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)


# Streamlit UI Customization (Hogwarts theme)
st.set_page_config(page_title="Hogwarts Q&A Chatbot", page_icon="🦉")
st.title("🦉 **Welcome to the Hogwarts Question-Answering Chatbot!** ⚡")
st.subheader("Embark on a magical journey with our wise and powerful wizarding assistants. Upload your magical texts and ask away! 📚✨")

session_id = "Default Session"


if 'store' not in st.session_state:
    st.session_state.store = {}


uploaded_files = st.file_uploader(label='🔮 Upload your magical scrolls (PDF files)', type='pdf', accept_multiple_files=True)


def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    return st.session_state.store[session_id]


# Whenever a document is uploaded it is split and embedded
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, 'wb') as file:
            file.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()
        documents.extend(docs)
    
    # Creating Splitter and Vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 500)
    splits = text_splitter.split_documents(documents)
    vectordb = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectordb.as_retriever()


    prompt_template = """
    Given a chat history and the latest user query, rewrite the query into a standalone question that is fully self-contained and does not rely on prior context. Do not answer the question—only rephrase or return it as is if no changes are needed.
    """

    prompt = ChatPromptTemplate(
        [
            ('system', prompt_template),
            MessagesPlaceholder('chat_history'),
            ('user', '{input}')
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)


    sys_prompt = (
    "You are a powerful wizard from the magical world of Hogwarts."
    "Your demeanor is proud and wise, yet you are not without wit."
    "Respond to the questions with your characteristic charm and intellect."
    "You may also throw in some magical references from the Harry Potter universe."
    "You are an assistant for answering questions in a fantastical and magical style."
    "Use the following pieces of context to answer the question."
    "If you do not know the answer, simply say: 'I do not know, young wizard.'"
    "\n\n"
    "{context}"
    )

    q_a_prompt = ChatPromptTemplate(
        [
            ("system", sys_prompt),
            MessagesPlaceholder("chat_history"),
            ('user', '{input}'),
        ]
    )


    question_ans_chain = create_stuff_documents_chain(llm, q_a_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_ans_chain)


    conversational_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    )


    user_input = st.text_input(label='🧙‍♂️ Enter your magical query:')
    if user_input:
        session_histtory = get_session_history(session_id=session_id)
        response = conversational_chain.invoke(
            {'input': user_input},
            config={'configurable': {'session_id': session_id}}
        )
        st.write("🦉 **Assistant**: ", response['answer'])
