import streamlit as st
from dotenv import load_dotenv
import os
import shelve
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from ragatouille import RAGPretrainedModel
from langchain.document_loaders import PyPDFLoader

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

load_dotenv()
file_path = "policy_data.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
full_document = ""
for page in pages:
    full_document += page.page_content

index_dir = ".ragatouille/colbert/indexes/policy_data/"
ivf_path = os.path.join(index_dir, "ivf.pid.pt")
index_exists = os.path.exists(ivf_path)
if not index_exists:
    st.write("Indexing document...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_document = "".join(page.page_content for page in pages)

    RAG.index(
        collection=[full_document],
        index_name="policy_data",
        max_document_length=512,
        split_documents=True,
    )
    retriever = RAG.as_langchain_retriever(k=3)
    st.write("Indexing complete!")
else:
    st.write("Loading existing index...")
    RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/policy_data/")
    retriever = RAG.as_langchain_retriever(k=3)

template = """Use the context below to answer the question.
Keep the answer concise and to the point.
If you are unsure about the answer, just say i do not know the answer to the question do not create your own answer and make sure the answer is concise and to the point.
Summarize the information such that main points are covered and if you think that there needs to be some more information added to the answer then you can add that information as well.
{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

chain_type_kwargs = {"prompt": prompt}
    

st.title("Streamlit Chatbot Interface")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, max_tokens=500, streaming=True)
chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
)

# Ensure openai_model is initialized in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"


# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        response = chain.invoke(prompt)
        full_response = response['result']
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
save_chat_history(st.session_state.messages)