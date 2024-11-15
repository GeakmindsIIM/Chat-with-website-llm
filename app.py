from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_vectorstore_from_url(url):
    #get the text in document form
    loader = WebBaseLoader(url)
    documents =  loader.load()

    #split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    # Create vector store from the chunks
    vector_store = FAISS.from_documents(document_chunks,OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([ 
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
        ])

    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's question based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain,stuff_documents_chain)

def get_response(user_query):
    retriever_chain = get_context_retriever_chain( st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
            "chat_history" : st.session_state.chat_history,
            "input" : user_input
        })
    return response['answer']
    


#app config
st.set_page_config(page_title="Chat with websites",page_icon="🤖")

st.title("Chat with websites")



#sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else :
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content = "Hello, I am a bot.How can I help you?")
            ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
    
    #user input
    user_input = st.chat_input("Type your message here...")
    if user_input is not None and user_input != "":
        response = get_response(user_input)
        st.session_state.chat_history.append(HumanMessage(content= user_input))
        st.session_state.chat_history.append(AIMessage(content= response))

    #conversation

    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)