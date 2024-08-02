import pandas as pd
import os
import re
import shutil
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import plotly.graph_objects as go
from chromadb.config import Settings
import json
import tempfile


st.set_page_config(page_title="Financial Advisor AI 💼", page_icon="💼", layout="wide")
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

openapi_key = st.secrets["OPENAI_API_KEY"]
if openapi_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
os.environ['OPENAI_API_KEY'] = openapi_key

# st.set_page_config(page_title="Financial Advisor AI 💼", page_icon="💼", layout="wide")




# Fetch API key from environment variable
# openapi_key = os.getenv('OPENAI_API_KEY')
# if openapi_key is None:
#     raise ValueError("OPENAI_API_KEY environment variable is not set")
# os.environ['OPENAI_API_KEY'] = openapi_key



# Configuration
class Config:
    DATA_FILE = 'Finance_data.csv'
    VECTOR_DB_DIR = tempfile.mkdtemp()
    
    # st.sidebar.markdown(f""" <h1 style="font-size:30px; color:#FFD700;">Quick Guide 🚀 </h1>""", unsafe_allow_html=True)
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


# Data Processing Functions
@st.cache_data
def load_and_process_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return [create_prompt_response(entry) for entry in tqdm(data.to_dict(orient='records'), desc="Processing data")]
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
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
    return [Document(page_content=f"Prompt: {entry['prompt']}\nResponse: {entry['response']}") for entry in prompt_response_data]

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)



@st.cache_resource
def create_vector_db(_texts, persist_directory):
    openai_embeddings = OpenAIEmbeddings()
    try:
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            # Load existing vector store
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=openai_embeddings)
            st.info("Loaded existing vector database.")
        else:
            # Create new vector store
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)  # Remove directory if it exists but is empty
            os.makedirs(persist_directory, exist_ok=True)  # Create directory
            vectordb = Chroma.from_documents(
                documents=_texts,
                embedding=openai_embeddings,
                persist_directory=persist_directory
            )
            st.success("Created new vector database.")
        return vectordb
    except Exception as e:
        st.error(f"An error occurred while creating/loading the vector database: {e}")
        return None






@st.cache_resource
def create_qa_chain(_vectordb):
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# User Profile Functions
def save_user_profile(profile):
    try:
        with open('user_profile.json', 'w') as f:
            json.dump(profile, f)
    except Exception as e:
        st.error(f"An error occurred while saving the user profile: {e}")

def load_user_profile():
    try:
        with open('user_profile.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the user profile: {e}")
        return None

# Investment Logic Functions
def calculate_risk_score(answers):
    score = sum(answers)
    if score <= 7:
        return "Low 🟢"
    elif score <= 14:
        return "Moderate 🟡"
    else:
        return "High 🔴"

def extract_allocation(result):
    lines = result.split('\n')
    allocation = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            if any(asset in key for asset in ['mutual funds', 'equity market', 'debentures', 'government bonds', 'fixed deposits', 'ppf', 'gold']):
                try:
                    allocation[key] = float(value.strip('%'))
                except ValueError:
                    pass
    return allocation

def suggest_rebalancing(current_portfolio, target_allocation):
    suggestions = []
    for asset, current_pct in current_portfolio.items():
        target_pct = target_allocation.get(asset, 0)
        diff = target_pct - current_pct
        if abs(diff) > 5:  # 5% threshold for rebalancing
            action = "increase" if diff > 0 else "decrease"
            suggestions.append(f"{action} {asset} by {abs(diff):.1f}%")
    return suggestions



# Visualization Function
def plot_allocation(allocation):
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#CCFF99']
    labels = list(allocation.keys())
    values = list(allocation.values())
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
    fig.update_layout(
        title='Investment Allocation 📊',
        font=dict(size=14),
        legend=dict(font=dict(size=12)),
        height=500
    )
    return fig


# Cache and Storage Management
def manage_cache_and_storage():
    if st.button("Clear Cache and Remove Vector Store"):
        try:
            # Clear Streamlit cache
            st.cache_data.clear()
            st.cache_resource.clear()
            # Remove vector store directory
            shutil.rmtree(Config.VECTOR_DB_DIR, ignore_errors=True)
            st.success("Cache cleared and vector store removed successfully.")
            # Remove the qa_chain from session state
            if 'qa_chain' in st.session_state:
                del st.session_state.qa_chain
        except Exception as e:
            st.error(f"An error occurred while clearing cache and removing vector store: {e}")



# Main Application
def main():
    # st.set_page_config(page_title="Financial Advisor AI 💼", page_icon="💼", layout="wide")
    
    st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size:40px;
        font-weight: bold;
        color: #43A047;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#4CAF50,#2E7D32);
    }
    .Widget>label {
        color: #FFFFFF;
        font-weight: bold;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #1E88E5;
        border-radius: 20px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


    # st.markdown('<p class="big-font">Financial Advisor AI 💼</p>', unsafe_allow_html=True)

    st.markdown(
    """
    <style>
    .box {
        background-color: #FFD700;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
        text-align: center;
    }
    .big-font {
        font-size: 50px;
        font-weight:900 ;
        color: #002F6C;
        margin-top:2px;
        margin-bottom:2px;
    }
    </style>
    <div class="box">
        <h1 class="big-font">AI Financial Advisor 💼</h1>
    </div>
    """,
    unsafe_allow_html=True
)

    
    sidebar_content()
    
    # Add cache and storage management
    manage_cache_and_storage()
    
    if st.button("🔄 Load Data"):
        load_data()
    
    if 'qa_chain' in st.session_state:
        qa_chain = st.session_state.qa_chain
        
        risk_score = risk_assessment()
        age, gender, purpose = user_profile()
        investment_query(qa_chain, age, gender, purpose, risk_score)
        portfolio_rebalancing()

def sidebar_content():
    st.sidebar.markdown('<hr style="border:1px solid #ff4b4b; margin-top:2px; margin-bottom:2px;">', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="subheader" >User Guide 🧭</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;margin-top:2px;">', unsafe_allow_html=True)
    
    # st.sidebar.markdown("### How to Use 📚")
    st.sidebar.markdown(f""" <h1 style="font-size:30px; color:#FFD700;">Quick Guide 🚀 </h1>""", unsafe_allow_html=True)
    st.sidebar.write(Config.HOW_TO_USE)
    
    st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
    # st.sidebar.markdown("### Sample Questions 💡")
    st.sidebar.markdown(f""" <h1 style="font-size:30px; color:#FFD700;">Sample Questions 💡 </h1>""", unsafe_allow_html=True)
    category = st.sidebar.selectbox("Choose a category:", list(Config.SAMPLE_QUESTIONS.keys()))
    selected_question = st.sidebar.selectbox("Select a sample question:", [""] + Config.SAMPLE_QUESTIONS[category])
    if selected_question:
        st.session_state.query = selected_question
    
    # Feedback Section
    st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <h1 style="font-size:30px; color:#FFD700;">Feedback 📝</h1>
    <p>We value your feedback! 😊</p>
        """, unsafe_allow_html=True)
    feedback = st.sidebar.slider("How helpful is this tool? 😞😐😊", 1, 5, 3)
    feedback_text = st.sidebar.text_area("Additional feedback:")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback! 👍")



   
def load_data():
    with st.spinner("Loading data... Please wait! 🕒"):
        try:
            data = load_and_process_data(Config.DATA_FILE)
            documents = create_documents(data)
            texts = split_documents(documents)
            vector_db = create_vector_db(texts, persist_directory=Config.VECTOR_DB_DIR)
            if vector_db:
                qa_chain = create_qa_chain(vector_db)
                st.session_state.qa_chain = qa_chain
                st.success("Data loaded successfully! 🎉")
            else:
                st.error("Failed to create vector database. 😕")
        except Exception as e:
            st.error(f"An error occurred while loading data: {str(e)} 😟")

def risk_assessment():
    st.markdown('<p class="subheader">Risk Assessment Questionnaire 📋</p>', unsafe_allow_html=True)
    questions = Config.RISK_ASSESSMENT_QUESTIONS
    answers = [st.slider(q, 1, 5, 3) for q in questions]
    
    risk_score = calculate_risk_score(answers)
    st.markdown(f"### Your risk tolerance is: {risk_score}")
    return risk_score

def user_profile():
    st.markdown('<p class="subheader">User Profile 👤</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age 🎂", min_value=18, max_value=100, value=30)
    with col2:
        gender = st.selectbox("Gender 🚻", ["Male", "Female", "Other"])
    with col3:
        purpose = st.text_input("Investment Purpose 🎯", "Retirement")
    return age, gender, purpose

def investment_query(qa_chain, age, gender, purpose, risk_score):
    st.markdown('<p class="subheader">Investment Query 💬</p>', unsafe_allow_html=True)
    query = st.text_area("Enter your investment question here:")
    
    if st.button("🚀 Get Advice"):
        if query:
            with st.spinner("Analyzing your query... 🤔"):
                try:
                    result = qa_chain.invoke({"query": query})["result"]
                    st.markdown("### AI Advice 🤖")
                    st.write(result)
                    
                    allocation = extract_allocation(result)
                    st.session_state.allocation = allocation
                    # fig = plot_allocation(allocation)
                    # st.plotly_chart(fig)
                    
                    user_profile = {
                        "age": age,
                        "gender": gender,
                        "purpose": purpose,
                        "risk_score": risk_score,
                        "allocation": allocation
                    }
                    save_user_profile(user_profile)
                    st.success("User profile saved! 💾")
                except Exception as e:
                    st.error(f"An error occurred while processing your query: {str(e)} 😟")
        else:
            st.warning("Please enter an investment question. ❓")

def portfolio_rebalancing():
    st.markdown('<p class="subheader">Portfolio Rebalancing 🔄</p>', unsafe_allow_html=True)
    if st.button("Suggest Rebalancing"):
        user_profile = load_user_profile()
        if user_profile and 'allocation' in st.session_state:
            current_portfolio = user_profile.get("allocation", {})
            target_allocation = st.session_state.allocation
            suggestions = suggest_rebalancing(current_portfolio, target_allocation)
            if suggestions:
                st.markdown("### Suggestions for rebalancing your portfolio:")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.success("Your portfolio is already well-balanced! 👍")
        else:
            st.warning("No user profile found. Please get advice first. ⚠️")

if __name__ == "__main__":
    main()




