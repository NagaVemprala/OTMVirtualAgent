import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, WebBaseLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# load_dotenv()

# Use the below option for cloud-based or web applications 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()
st.write("Getting the chat model ready ...")
chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
st.write("Initialization complete.")

# Configuration
MAIN_DOCS_FOLDER = 'Main_Docs'
CHROMA_DIR = 'chroma_db'
URL_CHROMA_DIR = 'url_chroma_db'
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(URL_CHROMA_DIR, exist_ok=True)

URLS = [
    "https://sites.up.edu/otm/otm-scholarships/",
    "https://business.up.edu/undergraduate/business-analytics.html",
    "https://business.up.edu/undergraduate/otm-bba.html",
    "https://up.smartcatalogiq.com/en/2023-2024/bulletin/dr-robert-b-pamplin-jr-school-of-business/degrees-and-programs/bachelor-of-business-administration-in-operation"
]

# Initialize session state
if 'question_asked' not in st.session_state:
    st.session_state.question_asked = False
if 'predefined_query' not in st.session_state:
    st.session_state.predefined_query = None

def get_chroma_db():
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def get_url_chroma_db():
    return Chroma(persist_directory=URL_CHROMA_DIR, embedding_function=embeddings)

def process_documents():
    """Process documents and URLs into separate Chroma DBs"""
    # Process Word documents
    if not os.listdir(CHROMA_DIR):
        with st.spinner("Processing documents..."):
            all_docs = []
            for filename in os.listdir(MAIN_DOCS_FOLDER):
                if filename.endswith('.docx'):
                    file_path = os.path.join(MAIN_DOCS_FOLDER, filename)
                    try:
                        loader = Docx2txtLoader(file_path)
                        documents = loader.load()
                        all_docs.extend(documents)
                        st.success(f"Processed document: {filename}")
                    except Exception as e:
                        st.error(f"Error processing {filename}: {e}")
            if all_docs:
                Chroma.from_documents(
                    documents=all_docs,
                    embedding=embeddings,
                    persist_directory=CHROMA_DIR
                )

    # Process URLs
    if not os.listdir(URL_CHROMA_DIR):
        with st.spinner("Processing URLs..."):
            all_url_docs = []
            for url in URLS:
                try:
                    loader = WebBaseLoader(url)
                    documents = loader.load()
                    all_url_docs.extend(documents)
                    st.success(f"Processed URL: {url}")
                except Exception as e:
                    st.error(f"Error processing {url}: {e}")
            if all_url_docs:
                Chroma.from_documents(
                    documents=all_url_docs,
                    embedding=embeddings,
                    persist_directory=URL_CHROMA_DIR
                )

# Streamlit UI
st.title("Welcome to University of Portland")
st.subheader("Supercool :violet[OTM virtual agent] using GenAI is here to chat with you! :sunglasses:")

# Process documents on startup
process_documents()

# Main question input
user_input = st.text_area("Enter your question!", "")
submit_btn = st.button("Submit")

# Predefined queries handling
query_mapping = {
    "scholarships": "Can you give me what scholarships are available",
    "career": "Can you tell me about career opportunities",
    "electives": "Can you provide information about electives"
}

# Show buttons only if no question has been asked
if not st.session_state.question_asked:
    st.write("Want to know more about specific information, click the buttons below:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Scholarships"):
            st.session_state.predefined_query = "scholarships"
            st.session_state.question_asked = True
    with col2:
        if st.button("Career Opportunities"):
            st.session_state.predefined_query = "career"
            st.session_state.question_asked = True
    with col3:
        if st.button("Electives"):
            st.session_state.predefined_query = "electives"
            st.session_state.question_asked = True

# Handle responses
def handle_query(query, db_selector):
    try:
        db = db_selector()
        qa = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3})
        )  # Add closing parenthesis here
        response = qa.invoke(query)['result']
        st.success("AI response --- ")
        st.write(response)
    except Exception as e:
        st.error(f"Query error: {str(e)}")


if submit_btn or st.session_state.predefined_query:
    if submit_btn and user_input.strip():
        st.session_state.question_asked = True
        handle_query(user_input, get_chroma_db)
    elif st.session_state.predefined_query:
        query_text = query_mapping[st.session_state.predefined_query]
        handle_query(query_text, get_url_chroma_db)
        st.session_state.predefined_query = None
    else:
        st.warning("Please enter a question.")