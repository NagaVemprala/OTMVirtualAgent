import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain_community.document_loaders import Docx2txtLoader, WebBaseLoader
from langchain.chains import RetrievalQA
import os

# load_dotenv()

# Use the below option for cloud-based or web applications 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings()
st.write("Getting the chat model ready ...")
chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
st.write("Initialization complete.")

# Configuration (Updated for FAISS)
MAIN_DOCS_FOLDER = 'Main_Docs'
FAISS_DIR = 'faiss_db'  # Changed to FAISS directory
URL_FAISS_DIR = 'url_faiss_db'
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(URL_FAISS_DIR, exist_ok=True)

URLS = [
    "https://sites.up.edu/otm/otm-scholarships/",
    "https://business.up.edu/undergraduate/business-analytics.html",
    "https://business.up.edu/undergraduate/otm-bba.html",
    "https://business.up.edu/undergraduate/majors.html",
    "https://up.smartcatalogiq.com/en/2023-2024/bulletin/dr-robert-b-pamplin-jr-school-of-business/degrees-and-programs/bachelor-of-business-administration-in-operations-and-technology-management/",
    "https://up.smartcatalogiq.com/en/2024-2025/bulletin/dr-robert-b-pamplin-jr-school-of-business/degrees-and-programs/bachelor-of-business-administration-in-supply-chain-analytics/",
    "https://business.up.edu/undergraduate/management-information-systems1.html"
]

def get_faiss_db():  # Updated function name
    return FAISS.load_local(FAISS_DIR, embeddings)  # Load FAISS from disk

def get_url_faiss_db():  # Updated function name
    return FAISS.load_local(URL_FAISS_DIR, embeddings)

def process_documents():
    """Process documents and URLs into separate FAISS DBs"""

    # Process Word documents
    if not os.listdir(FAISS_DIR):  # Check FAISS_DIR
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
                db = FAISS.from_documents(all_docs, embeddings)  # Create FAISS db
                db.save_local(FAISS_DIR)  # Save to disk

    # Process URLs
    if not os.listdir(URL_FAISS_DIR):  # Check URL_FAISS_DIR
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
                url_db = FAISS.from_documents(all_url_docs, embeddings)  # Create FAISS db
                url_db.save_local(URL_FAISS_DIR)  # Save to disk


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

# Handle responses (Updated for FAISS)
def handle_query(query, db_selector):
    try:
        db = db_selector()
        qa = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
        )
        response = qa.invoke(query)['result']
        st.success("AI response --- ")
        st.write(response)
    except Exception as e:
        st.error(f"Query error: {str(e)}")

if submit_btn or st.session_state.predefined_query:
    if submit_btn and user_input.strip():
        st.session_state.question_asked = True
        handle_query(user_input, get_faiss_db)  # Use FAISS function
    elif st.session_state.predefined_query:
        query_text = query_mapping[st.session_state.predefined_query]
        handle_query(query_text, get_url_faiss_db)  # Use FAISS function
        st.session_state.predefined_query = None
    else:
        st.warning("Please enter a question.")