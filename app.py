from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message

def load_data(urls):
    # Load data
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    return data

def split_documents(data):
    # Text Splitter
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    return docs

def initialize_chain(docs, temperature, k):
    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Vector Store
    vectorStore = FAISS.from_documents(docs, embeddings)
    # vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings)

    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": temperature, "truncation": True})
    QUESTION_PROMPT = PromptTemplate.from_template("""
        Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        This is a conversation with a human. Answer the questions you get based on the knowledge you have.
        If you don't know the answer, just say that you don't, don't try to make up an answer.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
        """)

    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,
                                               condense_question_prompt=QUESTION_PROMPT,
                                               return_source_documents=False, verbose=False)
    print(qa)
    return qa

def conversational_chat(qa, query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
def main():
    st.markdown("<h1 style='text-align: center;'>J.A.R.V.I.S</h1>", 
    unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Your Virtual Web Companion!</h3>", 
    unsafe_allow_html=True)

    st.write("Meet J.A.R.V.I.S: Your interactive browsing companion. Seamlessly engage with websites using AI. Explore, question, and navigate—transforming your browsing experience into conversations and discovery!")
  
    # Create sidebar for user inputs
    st.sidebar.title('Settings')
    option = st.sidebar.selectbox('Select Data Source', ('Web URL', 'Upload Document'))

    data  = []
    if option == 'Web URL':
        url_input = st.sidebar.text_input("Enter URLs (separate by commas)", key='url_input')
        if st.sidebar.button("Load Data"):
            urls_list = [url.strip() for url in url_input.split(",")]
            data = load_data(urls_list)
    else:
        file_upload = st.sidebar.file_uploader("Upload PDF", type='pdf')
        if file_upload:
            data = file_upload.read()

    docs = split_documents(data)

    temperature = st.sidebar.slider('Creativity (Temperature)', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    k_value = st.sidebar.select_slider('Number of Relevant Documents (K) ', options=[1, 2, 3], value=3)

    qa = initialize_chain(docs, temperature, k_value)

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I your personal bot powered by HuggingFace LLM 🤗. I can help you explore the document"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Enter question", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(qa, user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i))



if __name__ == "__main__":
    main()
