# import pandas as pd
# import os
# import logging
# from tqdm import tqdm
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import SKLearnVectorStore
# from sklearn.neighbors import NearestNeighbors
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA
# import streamlit as st
# import json
# import csv


# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# st.set_page_config(page_title="Financial Advisor AI 💼", page_icon="💼", layout="wide")

# # Load environment variables
# openapi_key = st.secrets.get("OPENAI_API_KEY")
# if not openapi_key:
#     logging.error("OPENAI_API_KEY is not set in Streamlit secrets")
#     st.error("OPENAI_API_KEY is not set. Please set it in your Streamlit secrets.")
# else:
#     os.environ['OPENAI_API_KEY'] = openapi_key

# class Config:
#     DATA_FILE = 'Finance_data.csv'
#     HOW_TO_USE = """
#     1. 📊 Click 'Load Data' to initialize the AI.
#     2. 📝 Complete the risk assessment questionnaire.
#     3. 👤 Fill in your age, gender, and investment purpose.
#     4. ❓ Enter your investment query in the text area.
#     5. 🚀 Click 'Get Advice' to receive personalized investment recommendations.
#     6. 📈 Review the advice and investment allocation chart.
#     7. 🔄 Use the portfolio rebalancing tool if needed.
#     8. 📚 Explore educational resources for more information.
#     """
#     SAMPLE_QUESTIONS = {
#         "Retirement 👴👵": [
#             "What's a good investment strategy for retirement in my 30s?",
#             "How should I adjust my retirement portfolio as I get closer to retirement age?"
#         ],
#         "Short-term Goals 🏠💍": [
#             "How should I invest for a down payment on a house in 5 years?",
#             "What are good investment options for saving for a wedding in 2 years?"
#         ],
#         "Long-term Growth 📈💰": [
#             "What's a good strategy for long-term wealth building?",
#             "How can I create a diversified portfolio for maximum growth over 20 years?"
#         ],
#         "Low-risk Options 🛡️💸": [
#             "What are some low-risk investment options for beginners?",
#             "How can I protect my savings from inflation with minimal risk?"
#         ],
#         "Tax-efficient Investing 📑💱": [
#             "What are the best options for tax-efficient investing?",
#             "How can I minimize my tax liability while maximizing returns?"
#         ]
#     }
#     RISK_ASSESSMENT_QUESTIONS = [
#         "On a scale of 1 to 5, how comfortable are you with taking risks in your investments? 😰😐😎",
#         "How would you react if your investment lost 10% of its value in a month? 😱😕🤔",
#         "How long do you plan to hold your investments before needing to access the funds? ⏱️💼",
#         "What is your primary goal for investing? 🎯💸"
#     ]




# @st.cache_data
# def load_and_process_data(file_path, chunk_size=1000):
#     try:
#         logging.info(f"Loading data from {file_path}")
#         st.info(f"Attempting to load data from {file_path}")
        
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file {file_path} does not exist.")
        
#         if not os.access(file_path, os.R_OK):
#             raise PermissionError(f"No read permission for the file {file_path}")
        
#         # Try different encodings
#         encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        
#         for encoding in encodings:
#             try:
#                 processed_data = []
#                 for chunk in pd.read_csv(file_path, encoding=encoding, chunksize=chunk_size):
#                     for _, row in chunk.iterrows():
#                         processed_data.append(create_prompt_response(row))
#                     st.text(f"Processed {len(processed_data)} entries so far...")
                
#                 st.success(f"Successfully loaded and processed {len(processed_data)} entries with {encoding} encoding.")
#                 return processed_data
#             except UnicodeDecodeError:
#                 continue
#             except Exception as e:
#                 st.error(f"Error processing data with {encoding} encoding: {str(e)}")
#                 continue
        
#         raise ValueError("Unable to read the file with any of the attempted encodings.")
    
#     except pd.errors.EmptyDataError:
#         error_msg = f"The file {file_path} is empty"
#         logging.error(error_msg)
#         st.error(error_msg)
#         return []
#     except IOError as e:
#         error_msg = f"IOError occurred while reading {file_path}: {e}"
#         logging.error(error_msg)
#         st.error(error_msg)
#         st.error(f"Current working directory: {os.getcwd()}")
#         st.error(f"File exists: {os.path.exists(file_path)}")
#         st.error(f"File is readable: {os.access(file_path, os.R_OK)}")
#         return []
#     except ValueError as e:
#         error_msg = f"ValueError: {str(e)}"
#         logging.error(error_msg)
#         st.error(error_msg)
#         return []
#     except Exception as e:
#         error_msg = f"An unexpected error occurred while loading data: {e}"
#         logging.error(error_msg)
#         st.error(error_msg)
#         return []

# def create_prompt_response(entry):
#     prompt = (
#         f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} "
#         f"for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
#     )
#     response = (
#         f"Based on your preferences, here are your investment options:\n"
#         f"- Mutual Funds: {entry['Mutual_Funds']}\n"
#         f"- Equity Market: {entry['Equity_Market']}\n"
#         f"- Debentures: {entry['Debentures']}\n"
#         f"- Government Bonds: {entry['Government_Bonds']}\n"
#         f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
#         f"- PPF: {entry['PPF']}\n"
#         f"- Gold: {entry['Gold']}\n"
#         f"Factors considered: {entry['Factor']}\n"
#         f"Objective: {entry['Objective']}\n"
#         f"Expected returns: {entry['Expect']}\n"
#         f"Investment monitoring: {entry['Invest_Monitor']}\n"
#         f"Reasons for choices:\n"
#         f"- Equity: {entry['Reason_Equity']}\n"
#         f"- Mutual Funds: {entry['Reason_Mutual']}\n"
#         f"- Bonds: {entry['Reason_Bonds']}\n"
#         f"- Fixed Deposits: {entry['Reason_FD']}\n"
#         f"Source of information: {entry['Source']}\n"
#     )
#     return {"prompt": prompt, "response": response}




# def create_documents(prompt_response_data):
#     logging.info(f"Creating {len(prompt_response_data)} documents")
#     return [Document(page_content=f"Prompt: {entry['prompt']}\nResponse: {entry['response']}") for entry in prompt_response_data]

# def split_documents(documents, chunk_size=1000, chunk_overlap=200):
#     logging.info(f"Splitting {len(documents)} documents")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     split_docs = text_splitter.split_documents(documents)
#     logging.info(f"Created {len(split_docs)} split documents")
#     return split_docs

# @st.cache_resource
# def create_vector_db(_texts):
#     logging.info("Creating vector database")
#     openai_embeddings = OpenAIEmbeddings()
#     try:
#         vectordb = SKLearnVectorStore.from_documents(
#             documents=_texts,
#             embedding=openai_embeddings,
#             algorithm="brute",  # You can also try "ball_tree" or "kd_tree"
#             n_neighbors=5
#         )
#         st.success("Created new vector database.")
#         return vectordb
#     except Exception as e:
#         logging.error(f"An error occurred while creating the vector database: {e}")
#         st.error(f"An error occurred while creating the vector database: {e}")
#         return None

# @st.cache_resource
# def create_qa_chain(_vectordb):
#     logging.info("Creating QA chain")
#     llm = OpenAI(temperature=0)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=_vectordb.as_retriever(),
#         return_source_documents=True
#     )
#     return qa_chain

# def save_user_profile(profile):
#     try:
#         logging.info("Saving user profile")
#         with open('user_profile.json', 'w') as f:
#             json.dump(profile, f)
#     except Exception as e:
#         logging.error(f"An error occurred while saving the user profile: {e}")
#         st.error(f"An error occurred while saving the user profile: {e}")

# def load_user_profile():
#     try:
#         logging.info("Loading user profile")
#         with open('user_profile.json', 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         logging.warning("No user profile found")
#         return None
#     except Exception as e:
#         logging.error(f"An error occurred while loading the user profile: {e}")
#         st.error(f"An error occurred while loading the user profile: {e}")
#         return None

# def calculate_risk_score(answers):
#     logging.info("Calculating risk score")
#     if len(answers) != 4:
#         raise ValueError("Expected 4 answers for the risk assessment")
#     try:
#         scores = list(map(int, answers))
#         return sum(scores) / len(scores)
#     except ValueError:
#         raise ValueError("Invalid input. Please provide numeric answers")

# def get_investment_advice(profile, question, qa_chain):
#     logging.info("Getting investment advice")
#     prompt = f"I'm a {profile['age']}-year-old {profile['gender']} looking to invest in {profile['Avenue']} " \
#              f"for {profile['Purpose']} over the next {profile['Duration']}. " \
#              f"My risk assessment score is {profile['risk_score']}. {question}"
#     response = qa_chain({"query": prompt})
#     return response["result"]

# def main():
#     st.title("Financial Advisor AI 💼")
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     if 'profile' not in st.session_state:
#         st.session_state.profile = {
#             "age": "",
#             "gender": "Male",
#             "Avenue": "",
#             "Purpose": "",
#             "Duration": "",
#             "risk_score": 0
#         }

#     st.sidebar.markdown("## Risk Assessment")
#     user_answers = []
#     for i, question in enumerate(Config.RISK_ASSESSMENT_QUESTIONS, 1):
#         answer = st.sidebar.radio(question, ['1', '2', '3', '4', '5'], key=f'question_{i}')
#         user_answers.append(answer)

#     if st.sidebar.button("Submit Risk Assessment"):
#         try:
#             risk_score = calculate_risk_score(user_answers)
#             st.session_state.profile["risk_score"] = risk_score
#             st.sidebar.success(f"Your risk score: {risk_score:.2f}")
#         except ValueError as e:
#             logging.error(f"Error in risk assessment: {e}")
#             st.sidebar.error(str(e))

#     st.sidebar.markdown("## User Profile")
#     st.session_state.profile["age"] = st.sidebar.text_input("Age", value=st.session_state.profile["age"])
#     st.session_state.profile["gender"] = st.sidebar.radio("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.profile["gender"]))
#     st.session_state.profile["Avenue"] = st.sidebar.text_input("Investment Avenue", value=st.session_state.profile["Avenue"])
#     st.session_state.profile["Purpose"] = st.sidebar.text_input("Investment Purpose", value=st.session_state.profile["Purpose"])
#     st.session_state.profile["Duration"] = st.sidebar.text_input("Investment Duration", value=st.session_state.profile["Duration"])

#     if st.sidebar.button("Save Profile"):
#         save_user_profile(st.session_state.profile)
#         st.sidebar.success("Profile saved successfully")

#     if st.sidebar.button("Load Profile"):
#         loaded_profile = load_user_profile()
#         if loaded_profile:
#             st.session_state.profile.update(loaded_profile)
#             st.sidebar.success("Profile loaded successfully")
#         else:
#             st.sidebar.error("No profile found")

#     st.markdown("## How to Use the App")
#     st.markdown(Config.HOW_TO_USE)

#     if st.button("Load Data"):
#         logging.info("Loading data")
#         prompt_response_data = load_and_process_data(Config.DATA_FILE)
#         if prompt_response_data:
#             documents = create_documents(prompt_response_data)
#             texts = split_documents(documents)
#             vector_db = create_vector_db(texts)
#             if vector_db:
#                 st.session_state.qa_chain = create_qa_chain(vector_db)
#                 st.success("Data loaded successfully. You can now ask your investment questions.")
#             else:
#                 st.error("Failed to create vector database.")
#         else:
#             st.error("Failed to load data")


    
#     st.markdown("## Investment Advice")
#     question = st.text_area("Enter your investment question here")

#     if st.button("Get Advice"):
#         if st.session_state.qa_chain:
#             logging.info("Getting investment advice")
#             advice = get_investment_advice(st.session_state.profile, question, st.session_state.qa_chain)
#             st.markdown("### Your Investment Advice")
#             st.write(advice)
#         else:
#             logging.warning("QA Chain is not available")
#             st.error("QA Chain is not available. Please load the data first.")

# if __name__ == "__main__":
#     logging.info("Starting the Financial Advisor AI application")
#     main()


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
import csv
import plotly.graph_objects as go
import plotly.express as px

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Page configuration
st.set_page_config(page_title="FinGenius: Your AI Financial Advisor", page_icon="💎", layout="wide")

# # Custom CSS with a cutting-edge, futuristic design
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Orbitron:wght@400;700&display=swap');
    
#     :root {
#         --primary-color: #00F5FF;
#         --secondary-color: #FF00E4;
#         --bg-color: #0A0E17;
#         --text-color: #E0E0E0;
#         --card-bg: #141C2F;
#     }
    
#     body {
#         color: var(--text-color);
#         background-color: var(--bg-color);
#         font-family: 'Roboto', sans-serif;
#         background-image: 
#             radial-gradient(circle at 10% 20%, rgba(0, 245, 255, 0.1) 0%, transparent 20%),
#             radial-gradient(circle at 90% 80%, rgba(255, 0, 228, 0.1) 0%, transparent 20%);
#         background-attachment: fixed;
#     }
    
#     .stApp {
#         background: transparent;
#     }
    
#     h1, h2, h3 {
#         font-family: 'Orbitron', sans-serif;
#         color: var(--primary-color);
#         text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
#     }
    
#     .stButton > button {
#         background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
#         color: var(--bg-color);
#         font-weight: 700;
#         border-radius: 30px;
#         border: none;
#         padding: 15px 30px;
#         font-size: 16px;
#         transition: all 0.3s ease;
#         box-shadow: 0 5px 15px rgba(0, 245, 255, 0.4);
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-3px) scale(1.05);
#         box-shadow: 0 8px 20px rgba(255, 0, 228, 0.6);
#     }
    
#     .stTextInput > div > div > input, 
#     .stSelectbox > div > div > select, 
#     .stTextArea > div > div > textarea {
#         background-color: var(--card-bg);
#         color: var(--text-color);
#         border-radius: 15px;
#         border: 2px solid var(--primary-color);
#         padding: 12px;
#         transition: all 0.3s ease;
#     }
    
#     .stTextInput > div > div > input:focus, 
#     .stSelectbox > div > div > select:focus, 
#     .stTextArea > div > div > textarea:focus {
#         border-color: var(--secondary-color);
#         box-shadow: 0 0 15px rgba(255, 0, 228, 0.5);
#     }
    
#     .stTab {
#         background-color: var(--card-bg);
#         color: var(--text-color);
#         font-weight: 600;
#         border-radius: 10px 10px 0 0;
#         border: 2px solid var(--primary-color);
#         border-bottom: none;
#         transition: all 0.3s ease;
#     }
    
#     .stTab[aria-selected="true"] {
#         background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
#         color: var(--bg-color);
#     }
    
#     .stDataFrame {
#         border: 2px solid var(--primary-color);
#         border-radius: 15px;
#         overflow: hidden;
#     }
    
#     .stDataFrame thead {
#         background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
#         color: var(--bg-color);
#     }
    
#     .stDataFrame tbody tr:nth-of-type(even) {
#         background-color: rgba(20, 28, 47, 0.7);
#     }
    
#     .stAlert {
#         background-color: var(--card-bg);
#         color: var(--text-color);
#         border-radius: 15px;
#         border: 2px solid var(--primary-color);
#     }
    
#     .stProgress > div > div > div > div {
#         background-color: var(--primary-color);
#     }
    
#     .stSlider > div > div > div > div {
#         color: var(--primary-color);
#     }
    
#     .css-1cpxqw2 {
#         background-color: var(--card-bg);
#         border-radius: 20px;
#         padding: 25px;
#         box-shadow: 0 10px 30px rgba(0, 245, 255, 0.2);
#         transition: all 0.3s ease;
#         border: 2px solid transparent;
#         background-clip: padding-box;
#     }
    
#     .css-1cpxqw2:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 15px 35px rgba(255, 0, 228, 0.3);
#         border-color: var(--secondary-color);
#     }
    
#     @keyframes glow {
#         0% { box-shadow: 0 0 5px var(--primary-color); }
#         50% { box-shadow: 0 0 20px var(--primary-color), 0 0 30px var(--secondary-color); }
#         100% { box-shadow: 0 0 5px var(--primary-color); }
#     }
    
#     .glow-effect {
#         animation: glow 2s infinite;
#     }
# </style>
# """, unsafe_allow_html=True)



# Updated Custom CSS with new fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;700&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --primary-color: #00F5FF;
        --secondary-color: #FF00E4;
        --bg-color: #0A0E17;
        --text-color: #E0E0E0;
        --card-bg: #141C2F;
    }
    
    body {
        color: var(--text-color);
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 245, 255, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(255, 0, 228, 0.1) 0%, transparent 20%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: transparent;
    }
    
    h1, h2, h3 {
        font-family: 'Exo 2', sans-serif;
        color: var(--primary-color);
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
        letter-spacing: 1px;
    }
    
    .stButton > button {
        font-family: 'Exo 2', sans-serif;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
        font-weight: 700;
        border-radius: 30px;
        border: none;
        padding: 15px 30px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 245, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 0, 228, 0.6);
    }
    
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select, 
    .stTextArea > div > div > textarea {
        font-family: 'Inter', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        border-radius: 15px;
        border: 2px solid var(--primary-color);
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus, 
    .stSelectbox > div > div > select:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: var(--secondary-color);
        box-shadow: 0 0 15px rgba(255, 0, 228, 0.5);
    }
    
    .stTab {
        font-family: 'Exo 2', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        font-weight: 600;
        border-radius: 10px 10px 0 0;
        border: 2px solid var(--primary-color);
        border-bottom: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stTab[aria-selected="true"] {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
    }
    
    .stDataFrame {
        font-family: 'Inter', sans-serif;
        border: 2px solid var(--primary-color);
        border-radius: 15px;
        overflow: hidden;
    }
    
    .stDataFrame thead {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
        font-family: 'Exo 2', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stDataFrame tbody tr:nth-of-type(even) {
        background-color: rgba(20, 28, 47, 0.7);
    }
    
    .stAlert {
        font-family: 'Inter', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        border-radius: 15px;
        border: 2px solid var(--primary-color);
    }
    
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
    
    .stSlider > div > div > div > div {
        color: var(--primary-color);
        font-family: 'Exo 2', sans-serif;
    }
    
    .css-1cpxqw2 {
        background-color: var(--card-bg);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0, 245, 255, 0.2);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    .css-1cpxqw2:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(255, 0, 228, 0.3);
        border-color: var(--secondary-color);
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px var(--primary-color); }
        50% { box-shadow: 0 0 20px var(--primary-color), 0 0 30px var(--secondary-color); }
        100% { box-shadow: 0 0 5px var(--primary-color); }
    }
    
    .glow-effect {
        animation: glow 2s infinite;
    }
</style>
""", unsafe_allow_html=True)




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
def load_and_process_data(file_path, chunk_size=1000):
    try:
        logging.info(f"Loading data from {file_path}")
        st.info(f"Attempting to load data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for the file {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                processed_data = []
                for chunk in pd.read_csv(file_path, encoding=encoding, chunksize=chunk_size):
                    for _, row in chunk.iterrows():
                        processed_data.append(create_prompt_response(row))
                    st.text(f"Processed {len(processed_data)} entries so far...")
                
                st.success(f"Successfully loaded and processed {len(processed_data)} entries with {encoding} encoding.")
                return processed_data
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Error processing data with {encoding} encoding: {str(e)}")
                continue
        
        raise ValueError("Unable to read the file with any of the attempted encodings.")
    
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
    except ValueError as e:
        error_msg = f"ValueError: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
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

def create_risk_profile_chart(risk_score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Profile", 'font': {'size': 24, 'color': '#E0E0E0'}},
        gauge = {
            'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "#E0E0E0"},
            'bar': {'color': "#3B82F6"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#E0E0E0",
            'steps': [
                {'range': [1, 2], 'color': '#10B981'},
                {'range': [2, 3], 'color': '#3B82F6'},
                {'range': [3, 4], 'color': '#F59E0B'},
                {'range': [4, 5], 'color': '#EF4444'}],
            'threshold': {
                'line': {'color': "#E0E0E0", 'width': 4},
                'thickness': 0.75,
                'value': risk_score}}))
    fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "#E0E0E0", 'family': "Poppins"})
    return fig

def create_investment_allocation_chart(advice):
    # This is a placeholder function. In a real scenario, you'd parse the advice
    # to extract allocation percentages. Here, we're using dummy data.
    labels = ['Stocks', 'Bonds', 'Real Estate', 'Cash']
    values = [40, 30, 20, 10]
    
    fig = px.pie(names=labels, values=values, title="Recommended Investment Allocation", hole=0.3)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0", family="Poppins"),
    )
    fig.update_traces(marker=dict(colors=['#3B82F6', '#10B981', '#F59E0B', '#EF4444']))
    return fig



def main():
    st.markdown('<div class="stHeader glow-effect"><h1 style="text-align: center;">FinGenius: Your AI Financial Advisor 💎</h1></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["👤 Profile & Risk", "💡 Investment Advice", "🎓 Learn"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("## 👤 User Profile")
            
            if 'profile' not in st.session_state:
                st.session_state.profile = {
                    "age": "",
                    "gender": "Male",
                    "Avenue": "",
                    "Purpose": "",
                    "Duration": "",
                    "risk_score": 0
                }
            
            st.session_state.profile["age"] = st.number_input("Age", min_value=18, max_value=100, value=int(st.session_state.profile["age"]) if st.session_state.profile["age"] else 30)
            st.session_state.profile["gender"] = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.profile["gender"]))
            st.session_state.profile["Avenue"] = st.selectbox("Investment Avenue", ["Stocks", "Bonds", "Real Estate", "Mutual Funds", "ETFs", "Cryptocurrencies"], index=0 if not st.session_state.profile["Avenue"] else ["Stocks", "Bonds", "Real Estate", "Mutual Funds", "ETFs", "Cryptocurrencies"].index(st.session_state.profile["Avenue"]))
            st.session_state.profile["Purpose"] = st.selectbox("Investment Purpose", ["Retirement", "Short-term Goals", "Long-term Growth", "Income Generation", "Capital Preservation"], index=0 if not st.session_state.profile["Purpose"] else ["Retirement", "Short-term Goals", "Long-term Growth", "Income Generation", "Capital Preservation"].index(st.session_state.profile["Purpose"]))
            st.session_state.profile["Duration"] = st.slider("Investment Duration (years)", 1, 30, value=int(st.session_state.profile["Duration"]) if st.session_state.profile["Duration"] else 10)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Profile"):
                    save_user_profile(st.session_state.profile)
                    st.success("Profile saved successfully!")
            with col2:
                if st.button("📂 Load Profile"):
                    loaded_profile = load_user_profile()
                    if loaded_profile:
                        st.session_state.profile.update(loaded_profile)
                        st.success("Profile loaded successfully!")
                    else:
                        st.error("No profile found.")
        
        with col2:
            st.sidebar.markdown("## 📈 Risk Assessment")
            user_answers = []
            for i, question in enumerate(Config.RISK_ASSESSMENT_QUESTIONS, 1):
                answer = st.sidebar.select_slider(question, options=['1', '2', '3', '4', '5'], key=f'question_{i}')
                user_answers.append(answer)
            
            if st.sidebar.button("📊 Calculate Risk Profile"):
                try:
                    risk_score = calculate_risk_score(user_answers)
                    st.session_state.profile["risk_score"] = risk_score
                    st.sidebar.success(f"Your risk score: {risk_score:.2f}")
                    st.sidebar.plotly_chart(create_risk_profile_chart(risk_score), use_container_width=True)
                except ValueError as e:
                    logging.error(f"Error in risk assessment: {e}")
                    st.sidebar.error(str(e))
    
    with tab2:
        st.markdown("## 💡 Investment Advice")
        
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None
        
        if st.button("🔄 Load Financial Data"):
            with st.spinner("Loading data..."):
                logging.info("Loading data")
                prompt_response_data = load_and_process_data(Config.DATA_FILE)
                if prompt_response_data:
                    documents = create_documents(prompt_response_data)
                    texts = split_documents(documents)
                    vector_db = create_vector_db(texts)
                    if vector_db:
                        st.session_state.qa_chain = create_qa_chain(vector_db)
                        st.success("Data loaded successfully. You can now ask for investment advice!")
                    else:
                        st.error("Failed to create vector database.")
                else:
                    st.error("Failed to load data.")
        
        question = st.text_area("💬 What would you like to know about investing?", height=100)
        
        if st.button("🚀 Get Personalized Advice"):
            if st.session_state.qa_chain:
                with st.spinner("Generating your personalized investment advice..."):
                    logging.info("Getting investment advice")
                    advice = get_investment_advice(st.session_state.profile, question, st.session_state.qa_chain)
                    st.markdown("### 🎯 Your Personalized Investment Advice")
                    st.info(advice)
                    
                    # Display investment allocation chart
                    st.plotly_chart(create_investment_allocation_chart(advice), use_container_width=True)
            else:
                logging.warning("QA Chain is not available")
                st.error("Please load the financial data first by clicking the 'Load Financial Data' button.")
    
    with tab3:
        st.markdown("## 🎓 Financial Education Hub")
        st.markdown("### 🤔 Sample Questions to Get You Started")
        for category, questions in Config.SAMPLE_QUESTIONS.items():
            with st.expander(category):
                for q in questions:
                    st.write(f"• {q}")
        
        st.markdown("### 📚 Educational Resources")
        with st.expander("📘 Investment Basics"):
            st.write("Learn the fundamental concepts of investing, including asset classes, risk vs. return, and diversification.")
        with st.expander("🛡️ Risk Management Strategies"):
            st.write("Discover techniques to manage and mitigate investment risks, including portfolio diversification and hedging strategies.")
        with st.expander("💰 Tax-Efficient Investing"):
            st.write("Explore strategies to minimize your tax liability while maximizing your investment returns.")
        with st.expander("📅 Retirement Planning"):
            st.write("Learn how to plan and save effectively for your retirement, including information on 401(k)s, IRAs, and other retirement accounts.")
        with st.expander("📈 Market Analysis Techniques"):
            st.write("Discover various methods for analyzing financial markets, including fundamental and technical analysis.")

if __name__ == "__main__":
    logging.info("Starting FinGenius: Your AI Financial Advisor")
    main()





