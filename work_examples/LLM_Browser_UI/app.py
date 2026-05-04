import streamlit as st
import numpy as np
import os
import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- RAG IMPORTS and CONSTANTS ---
# NOTE: Changed import from UnstructuredPDFLoader to PyPDFLoader to fix dependency conflict
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ollama # Necessary for explicit model pulling

# NOTE: These models must be pulled locally using `ollama pull model_name`
EMBED_MODEL = "nomic-embed-text" 
LLM_MODEL = "llama3.2" 
RAG_COLLECTION_NAME = "rag_pdf_collection"

# Global variable to hold the initialized RAG chain
RAG_CHAIN = None


# --- 1. IRIS Model Training & Setup ---
@st.cache_resource
def load_and_train_iris_model():
    """Loads the Iris dataset and trains a Decision Tree Classifier."""
    try:
        # Load Iris data
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        
        # Split data
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )
        
        # Train model
        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X_train, y_train)
        
        return clf, iris.target_names, iris.feature_names
    except Exception as e:
        print(f"FATAL IRIS MODEL ERROR during startup: {e}")
        return None, None, None

clf, class_names, feature_names = load_and_train_iris_model()


# --- RAG Setup Functions ---

def load_pdfs_from_directory(path):
    """Loads all PDF and TXT documents from a given directory path."""
    if not os.path.exists(path):
        return []

    # Load PDF files
    # Using PyPDFLoader to avoid complex unstructured dependency issues
    pdf_loader = DirectoryLoader(
        path=path,
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader, # CHANGED from UnstructuredPDFLoader
        use_multithreading=True
    )
    pdf_documents = pdf_loader.load()

    # Load TXT files (using glob for txt files)
    txt_documents = []
    for filepath in glob.glob(os.path.join(path, '**', '*.txt'), recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Represent text file as a LangChain Document
                txt_documents.append({"page_content": content, "metadata": {"source": filepath}})
        except Exception as e:
            print(f"Could not load text file {filepath}: {e}")

    # Combine all loaded documents
    all_documents = pdf_documents # Langchain PDF loader returns list of Document objects
    
    # Simple document creation for TXT content
    from langchain.docstore.document import Document
    for doc_dict in txt_documents:
        all_documents.append(Document(page_content=doc_dict['page_content'], metadata=doc_dict['metadata']))

    return all_documents


@st.cache_resource
def build_rag_chain(_documents, embed_model=EMBED_MODEL, llm_model=LLM_MODEL):
    """Initializes and returns the complete RAG chain."""
    
    if not _documents:
        return None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = splitter.split_documents(_documents)

    try:
        # 1. Ensure Ollama models are available
        ollama.pull(embed_model)
        ollama.pull(llm_model)

        # 2. Setup Vector Store (Chroma)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=embed_model),
            collection_name=RAG_COLLECTION_NAME,
        )

        llm = ChatOllama(model=llm_model)

        # 3. Setup MultiQuery Retriever
        multi_query_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are an AI assistant. Generate 5 different versions of the user question "
                "to improve document retrieval from a vector database.\n"
                "Original question: {question}"
            ),
        )

        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(),
            llm=llm,
            prompt=multi_query_prompt,
        )

        # 4. Setup Contextual Prompt and Chain
        context_prompt = ChatPromptTemplate.from_template(
            "Answer the question using ONLY the context below:\n{context}\n\nQuestion: {question}"
        )

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | context_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        print(f"RAG CHAIN BUILD ERROR: {e}")
        return None

def execute_rag_upload(directory_path: str):
    """Executes the file loading and RAG chain building process."""
    global RAG_CHAIN
    
    documents = load_pdfs_from_directory(directory_path)

    if not documents:
        return False, f"No PDF or TXT files found in: {directory_path}."

    # Build the RAG chain using the loaded documents
    RAG_CHAIN = build_rag_chain(documents)
    
    if RAG_CHAIN:
        file_count = len(documents)
        return True, f"Successfully loaded {file_count} documents, split, embedded, and RAG chain built."
    else:
        return False, "Failed to initialize RAG chain (Check Ollama models)."

def execute_rag_query(query: str):
    """Executes the RAG chain query."""
    global RAG_CHAIN
    
    if not query.strip():
        return False, "Query cannot be empty."
        
    if RAG_CHAIN is None:
        return False, "RAG system is not initialized. Please upload files first."

    try:
        # Invoke the RAG chain
        response = RAG_CHAIN.invoke(query)
        return True, response
    except Exception as e:
        return False, f"Ollama/RAG execution failed: {e}"


# --- 3. Input Validation Functions (Required for assignment) ---

def validate_directory(path: str) -> tuple[bool, str]:
    """Validates that the path is an existing directory ('No bad directories')."""
    if not os.path.isdir(path):
        return False, "Error: Path is not a valid directory or does not exist."
    return True, "Directory is valid."

def validate_iris_inputs(data: list[str]) -> tuple[bool, str, list[float]]:
    """Validates IRIS parameters (4 float values, positive numbers - 'No bad inputs')."""
    if len(data) != 4:
        return False, "Error: Four parameters must be entered.", []
    
    validated_data = []
    for i, val in enumerate(data):
        try:
            f_val = float(val)
            # Validation for 'No bad inputs' (ensure positivity and reasonable size)
            if f_val <= 0 or f_val > 10: # Assuming 10cm is a safe upper bound
                return False, f"Error: {feature_names[i]} must be a positive number and less than 10cm.", []
            validated_data.append(f_val)
        except ValueError:
            return False, f"Error: All four inputs must be valid numbers. Check value for {feature_names[i]}.", []

    return True, "Inputs are valid.", validated_data


# --- 4. Streamlit GUI Setup ---

st.set_page_config(page_title="AICC-220 Final Assignment")
st.title("Unified AI Application Interface (RAG & IRIS)")

# Add a general warning if the IRIS model failed to load
if clf is None:
     st.error("FATAL: IRIS Classification functionality is disabled because the model failed to load. Please ensure all required dependencies (like scikit-learn) are installed correctly.")


# Initialize session state for task selection
if 'current_task' not in st.session_state:
    st.session_state.current_task = 'none'

# --- Task Selection Buttons (Sidebar) ---
st.sidebar.header("Select Task")

def set_task(task_name):
    st.session_state.current_task = task_name

st.sidebar.button("1. RAG File Upload", on_click=set_task, args=['rag_upload'])
st.sidebar.button("2. RAG Query", on_click=set_task, args=['rag_query'])
st.sidebar.button("3. IRIS Classification", on_click=set_task, args=['iris_classify'])

st.divider()

# --- Conditional Rendering of Forms/Tasks ---

if st.session_state.current_task == 'none':
    st.info("Click a button in the sidebar to begin one of the three required tasks.")

# ----------------------------------------------------------------------
## RAG File Upload
elif st.session_state.current_task == 'rag_upload':
    st.header("1. Choose Directory with PDF/TXT Files")
    
    if RAG_CHAIN is not None:
        st.success("RAG system is already initialized and ready for queries!")
    
    with st.form("rag_upload_form"):
        # Text input for the directory path
        upload_path = st.text_input(
            "Enter Directory Path (containing .txt or .pdf files):",
            placeholder="e.g., C:/data/docs"
        )
        submitted = st.form_submit_button("Upload Files and Initialize RAG")

        if submitted:
            # 1. Validation: Check for 'No bad directories'
            is_valid, message = validate_directory(upload_path)
            
            if is_valid:
                st.info(f"Directory check passed: {message}. Starting upload and chain build...")
                # 2. Execution (REAL RAG INGESTION)
                success, upload_message = execute_rag_upload(upload_path)
                
                if success:
                    st.success(f"Upload Complete: {upload_message}")
                    st.session_state.current_task = 'rag_query' # Switch to query view on success
                    st.rerun()
                else:
                    st.error(f"Upload Failed: {upload_message}. Check Ollama models and dependencies.")
            else:
                st.error(f"Validation Failed: {message}")

# ----------------------------------------------------------------------
## RAG/Ollama Query
elif st.session_state.current_task == 'rag_query':
    st.header("2. Make a Query to Ollama/RAG")
    
    if RAG_CHAIN is None:
        st.warning("RAG system is not initialized. Please go to '1. RAG File Upload' first.")
    
    with st.form("rag_query_form"):
        query = st.text_area(
            "Enter your question for the RAG system:",
            placeholder="e.g., What are the key findings from the uploaded documents?",
            disabled=RAG_CHAIN is None
        )
        submitted = st.form_submit_button("Submit Query", disabled=RAG_CHAIN is None)

        if submitted:
            # 1. Validation (Minimal check for query content)
            if not query.strip():
                st.error("Please enter a question.")
            else:
                st.info("Sending query to RAG system (Ollama)...")
                
                # 2. Execution (REAL RAG QUERY)
                with st.spinner("Retrieving context and generating response..."):
                    success, response = execute_rag_query(query)
                
                if success:
                    st.subheader("RAG Response:")
                    st.success(response)
                else:
                    st.error(f"Query Failed: {response}")

# ----------------------------------------------------------------------
## IRIS Classification
elif st.session_state.current_task == 'iris_classify':
    st.header("3. IRIS Species Classification")
    st.markdown("Enter the four required parameters (in cm) for classification.")
    
    if clf is None:
        # If model failed to load during startup, show an explicit error here too.
        st.warning("IRIS model not loaded or trained correctly. Cannot classify.")
    else:
        with st.form("iris_form"):
            # Inputs for the four features
            col1, col2 = st.columns(2)
            
            sepal_length = col1.text_input(f"{feature_names[0]} (cm)", placeholder="e.g., 5.1")
            sepal_width = col2.text_input(f"{feature_names[1]} (cm)", placeholder="e.g., 3.5")
            
            col3, col4 = st.columns(2)
            petal_length = col3.text_input(f"{feature_names[2]} (cm)", placeholder="e.g., 1.4")
            petal_width = col4.text_input(f"{feature_names[3]} (cm)", placeholder="e.g., 0.2")

            submitted = st.form_submit_button("Get Species Prediction")

            if submitted:
                input_data = [sepal_length, sepal_width, petal_length, petal_width]
                
                # 1. Validation: Check for 'No bad inputs for IRIS'
                is_valid, message, final_data = validate_iris_inputs(input_data)
                
                if is_valid:
                    st.success(f"Input Data Valid: {message}")
                    st.info("Running Decision Tree Classifier...")
                    
                    # 2. Execution
                    # The model expects a 2D array: [[sepal_L, sepal_W, petal_L, petal_W]]
                    input_array = np.array([final_data])
                    
                    try:
                        # Predict the class index
                        y_pred = clf.predict(input_array)
                        # Get the class name
                        predicted_species = class_names[y_pred[0]]
                        
                        st.subheader("Classification Result")
                        st.success(f"Predicted Species: **{predicted_species}**")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                else:
                    st.error(f"Validation Failed: {message}")
