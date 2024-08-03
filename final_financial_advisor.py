import pandas as pd
import os
import logging
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SKLearnVectorStore
from sklearn.neighbors import NearestNeighbors
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Financial Advisor AI 💼", page_icon="💼", layout="wide")

# Load environment variables
openapi_key = st.secrets.get("OPENAI_API_KEY")
if not openapi_key:
    logging.error("OPENAI_API_KEY is not set in Streamlit secrets")
    st.error("OPENAI_API_KEY is not set. Please set it in your Streamlit secrets.")
else:
    os.environ['OPENAI_API_KEY'] = openapi_key

class Config:
    DATA_FILE = 'Finance_data.csv'
    HOW_TO_USE = """
    1. 📊 Click 'Load Data' to initialize the AI.
    2. 📝 Complete the risk assessment questionnaire.
    3. 👤 Fill in your age, gender, and investment purpose.
    4. ❓ Enter your investment query in the text area.
    5. 🚀 Click 'Get Advice' to receive personalized investment recommendations.
    6. 📈 Review the advice and investment allocation chart.
    7. 🔄 Use the portfolio rebalancing tool if needed.
    8. 📚 Explore educational resources for more information.
    """
    SAMPLE_QUESTIONS = {
        "Retirement 👴👵": [
            "What's a good investment strategy for retirement in my 30s?",
            "How should I adjust my retirement portfolio as I get closer to retirement age?"
        ],
        "Short-term Goals 🏠💍": [
            "How should I invest for a down payment on a house in 5 years?",
            "What are good investment options for saving for a wedding in 2 years?"
        ],
        "Long-term Growth 📈💰": [
            "What's a good strategy for long-term wealth building?",
            "How can I create a diversified portfolio for maximum growth over 20 years?"
        ],
        "Low-risk Options 🛡️💸": [
            "What are some low-risk investment options for beginners?",
            "How can I protect my savings from inflation with minimal risk?"
        ],
        "Tax-efficient Investing 📑💱": [
            "What are the best options for tax-efficient investing?",
            "How can I minimize my tax liability while maximizing returns?"
        ]
    }
    RISK_ASSESSMENT_QUESTIONS = [
        "On a scale of 1 to 5, how comfortable are you with taking risks in your investments? 😰😐😎",
        "How would you react if your investment lost 10% of its value in a month? 😱😕🤔",
        "How long do you plan to hold your investments before needing to access the funds? ⏱️💼",
        "What is your primary goal for investing? 🎯💸"
    ]

@st.cache_data
def load_and_process_data(file_path):
    try:
        logging.info(f"Loading data from {file_path}")
        st.info(f"Attempting to load data from {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for the file {file_path}")
        
        # Try to read the file
        data = pd.read_csv(file_path)
        
        processed_data = [create_prompt_response(entry) for entry in tqdm(data.to_dict(orient='records'), desc="Processing data")]
        logging.info(f"Processed {len(processed_data)} entries")
        st.success(f"Successfully loaded and processed {len(processed_data)} entries")
        return processed_data
    except pd.errors.EmptyDataError:
        error_msg = f"The file {file_path} is empty"
        logging.error(error_msg)
        st.error(error_msg)
        return []
    except IOError as e:
        error_msg = f"IOError occurred while reading {file_path}: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"File exists: {os.path.exists(file_path)}")
        st.error(f"File is readable: {os.access(file_path, os.R_OK)}")
        return []
    except Exception as e:
        error_msg = f"An unexpected error occurred while loading data: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        return []

def create_prompt_response(entry):
    prompt = (
        f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} "
        f"for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
    )
    response = (
        f"Based on your preferences, here are your investment options:\n"
        f"- Mutual Funds: {entry['Mutual_Funds']}\n"
        f"- Equity Market: {entry['Equity_Market']}\n"
        f"- Debentures: {entry['Debentures']}\n"
        f"- Government Bonds: {entry['Government_Bonds']}\n"
        f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
        f"- PPF: {entry['PPF']}\n"
        f"- Gold: {entry['Gold']}\n"
        f"Factors considered: {entry['Factor']}\n"
        f"Objective: {entry['Objective']}\n"
        f"Expected returns: {entry['Expect']}\n"
        f"Investment monitoring: {entry['Invest_Monitor']}\n"
        f"Reasons for choices:\n"
        f"- Equity: {entry['Reason_Equity']}\n"
        f"- Mutual Funds: {entry['Reason_Mutual']}\n"
        f"- Bonds: {entry['Reason_Bonds']}\n"
        f"- Fixed Deposits: {entry['Reason_FD']}\n"
        f"Source of information: {entry['Source']}\n"
    )
    return {"prompt": prompt, "response": response}

def create_documents(prompt_response_data):
    logging.info(f"Creating {len(prompt_response_data)} documents")
    return [Document(page_content=f"Prompt: {entry['prompt']}\nResponse: {entry['response']}") for entry in prompt_response_data]

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    logging.info(f"Splitting {len(documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Created {len(split_docs)} split documents")
    return split_docs

@st.cache_resource
def create_vector_db(_texts):
    logging.info("Creating vector database")
    openai_embeddings = OpenAIEmbeddings()
    try:
        vectordb = SKLearnVectorStore.from_documents(
            documents=_texts,
            embedding=openai_embeddings,
            algorithm="brute",  # You can also try "ball_tree" or "kd_tree"
            n_neighbors=5
        )
        st.success("Created new vector database.")
        return vectordb
    except Exception as e:
        logging.error(f"An error occurred while creating the vector database: {e}")
        st.error(f"An error occurred while creating the vector database: {e}")
        return None

@st.cache_resource
def create_qa_chain(_vectordb):
    logging.info("Creating QA chain")
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def save_user_profile(profile):
    try:
        logging.info("Saving user profile")
        with open('user_profile.json', 'w') as f:
            json.dump(profile, f)
    except Exception as e:
        logging.error(f"An error occurred while saving the user profile: {e}")
        st.error(f"An error occurred while saving the user profile: {e}")

def load_user_profile():
    try:
        logging.info("Loading user profile")
        with open('user_profile.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("No user profile found")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading the user profile: {e}")
        st.error(f"An error occurred while loading the user profile: {e}")
        return None

def calculate_risk_score(answers):
    logging.info("Calculating risk score")
    if len(answers) != 4:
        raise ValueError("Expected 4 answers for the risk assessment")
    try:
        scores = list(map(int, answers))
        return sum(scores) / len(scores)
    except ValueError:
        raise ValueError("Invalid input. Please provide numeric answers")

def get_investment_advice(profile, question, qa_chain):
    logging.info("Getting investment advice")
    prompt = f"I'm a {profile['age']}-year-old {profile['gender']} looking to invest in {profile['Avenue']} " \
             f"for {profile['Purpose']} over the next {profile['Duration']}. " \
             f"My risk assessment score is {profile['risk_score']}. {question}"
    response = qa_chain({"query": prompt})
    return response["result"]

def main():
    st.title("Financial Advisor AI 💼")
    
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    if 'profile' not in st.session_state:
        st.session_state.profile = {
            "age": "",
            "gender": "Male",
            "Avenue": "",
            "Purpose": "",
            "Duration": "",
            "risk_score": 0
        }

    st.sidebar.markdown("## Risk Assessment")
    user_answers = []
    for i, question in enumerate(Config.RISK_ASSESSMENT_QUESTIONS, 1):
        answer = st.sidebar.radio(question, ['1', '2', '3', '4', '5'], key=f'question_{i}')
        user_answers.append(answer)

    if st.sidebar.button("Submit Risk Assessment"):
        try:
            risk_score = calculate_risk_score(user_answers)
            st.session_state.profile["risk_score"] = risk_score
            st.sidebar.success(f"Your risk score: {risk_score:.2f}")
        except ValueError as e:
            logging.error(f"Error in risk assessment: {e}")
            st.sidebar.error(str(e))

    st.sidebar.markdown("## User Profile")
    st.session_state.profile["age"] = st.sidebar.text_input("Age", value=st.session_state.profile["age"])
    st.session_state.profile["gender"] = st.sidebar.radio("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.profile["gender"]))
    st.session_state.profile["Avenue"] = st.sidebar.text_input("Investment Avenue", value=st.session_state.profile["Avenue"])
    st.session_state.profile["Purpose"] = st.sidebar.text_input("Investment Purpose", value=st.session_state.profile["Purpose"])
    st.session_state.profile["Duration"] = st.sidebar.text_input("Investment Duration", value=st.session_state.profile["Duration"])

    if st.sidebar.button("Save Profile"):
        save_user_profile(st.session_state.profile)
        st.sidebar.success("Profile saved successfully")

    if st.sidebar.button("Load Profile"):
        loaded_profile = load_user_profile()
        if loaded_profile:
            st.session_state.profile.update(loaded_profile)
            st.sidebar.success("Profile loaded successfully")
        else:
            st.sidebar.error("No profile found")

    st.markdown("## How to Use the App")
    st.markdown(Config.HOW_TO_USE)

    if st.button("Load Data"):
        logging.info("Loading data")
        prompt_response_data = load_and_process_data(Config.DATA_FILE)
        if prompt_response_data:
            documents = create_documents(prompt_response_data)
            texts = split_documents(documents)
            vector_db = create_vector_db(texts)
            if vector_db:
                st.session_state.qa_chain = create_qa_chain(vector_db)
                st.success("Data loaded successfully. You can now ask your investment questions.")
            else:
                st.error("Failed to create vector database.")
        else:
            st.error("Failed to load data")

    st.markdown("## Investment Advice")
    question = st.text_area("Enter your investment question here")

    if st.button("Get Advice"):
        if st.session_state.qa_chain:
            logging.info("Getting investment advice")
            advice = get_investment_advice(st.session_state.profile, question, st.session_state.qa_chain)
            st.markdown("### Your Investment Advice")
            st.write(advice)
        else:
            logging.warning("QA Chain is not available")
            st.error("QA Chain is not available. Please load the data first.")

if __name__ == "__main__":
    logging.info("Starting the Financial Advisor AI application")
    main()
