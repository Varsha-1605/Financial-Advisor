import streamlit as st
import pandas as pd
import os
import logging
from tqdm import tqdm
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import SKLearnVectorStore
from langchain_community.vectorstores import SKLearnVectorStore
from sklearn.neighbors import NearestNeighbors
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import json
import plotly.graph_objects as go
import plotly.express as px
from faker import Faker
import random
from langchain_openai import ChatOpenAI
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from langchain.prompts import PromptTemplate
import time
from openai import OpenAIError, APIConnectionError
from requests.exceptions import ConnectionError
import traceback
import sys
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Page configuration
# Encode image to base64
with open("New_logo.png", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
# st.set_page_config(page_title="MarketWealth Genius: Your AI Financial Advisor", page_icon="💎", layout="wide")

# Custom CSS (unchanged)
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
    
    /* ... (all your existing CSS rules) ... */

    .glow-effect {
        animation: glow 2s infinite;
    }

    /* Add the new link styles here */
    a {
        color: var(--primary-color);
        text-decoration: none;
        transition: all 0.3s ease;
        position: relative;
    }

    a:hover {
        color: var(--secondary-color);
    }

    a::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 2px;
        bottom: -2px;
        left: 0;
        background-color: var(--secondary-color);
        visibility: hidden;
        transform: scaleX(0);
        transition: all 0.3s ease-in-out;
    }

    a:hover::after {
        visibility: visible;
        transform: scaleX(1);
    }
            

</style>
""", unsafe_allow_html=True)

#

# Add responsive header
# Add responsive header
st.markdown(
    f'''
    <style>
    .header {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 20px;
    }}
    .header img {{
        max-width: 100%;
        height: auto;
        width: 80px; /* Default width */
    }}
    @media (min-width: 600px) {{
        .header h1 {{
            font-size: 2.5em;
        }}
        .header img {{
            width: 100px; /* Image size for wider screens */
        }}
    }}
    @media (max-width: 599px) {{
        .header h1 {{
            font-size: 1.8em;
        }}
        .header img {{
            width: 80px; /* Image size for smaller screens */
        }}
    }}
    </style>
    <div class="header">
        <img src="data:image/png;base64,{image_base64}" />
        <h1>MarketWealth Genius: Your AI Financial Advisor</h1>
    </div>
    ''',
    unsafe_allow_html=True
)

# Load environment variables
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("OPENAI_API_KEY is not set in Streamlit secrets")
    st.error("OPENAI_API_KEY is not set. Please set it in your Streamlit secrets.")
else:
    os.environ['OPENAI_API_KEY'] = openai_api_key

# Configuration class (unchanged)
class Config:
    DATA_FILE = 'Finance_data.csv'
    HOW_TO_USE = """
    ⬇️ Scroll down and click 'BEGIN YOUR INVESTMENT JOURNEY'.
    📊 Click 'Load Data' to initialize the AI.
    📝 Complete the risk assessment questionnaire.
    👤 Fill in your profile information.
    ❓ Enter your investment query in the text area.
    🚀 Click 'Get Advice' to receive personalized investment recommendations.
    📈 Review the advice and investment allocation chart.
    🔄 Use the portfolio rebalancing tool if needed.
    📚 Explore educational resources for more information.
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
        "Low-risk Options 🛡💸": [
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
        "How long do you plan to hold your investments before needing to access the funds? ⏱💼",
        "What is your primary goal for investing? 🎯💸"
    ]

    FD_RATES = {
        "Unity Small Finance Bank": 9.0,
        "Utkarsh Small Finance Bank": 8.5,
        "RBL Bank": 8.1,
        "SBM Bank India": 8.25,
        "Bandhan Bank": 8.0
    }
    FD_SENIOR_CITIZEN_RATE_PREMIUM = 0.5
# Data generation functions (unchanged)
fake = Faker()

def generate_customer_data():
    # ... (unchanged)
    age = random.randint(20, 70)
    gender = random.choice(['Male', 'Female'])
    marital_status = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
    income_level = random.choice(['Low', 'Medium', 'High'])
    education = random.choice(['High School', 'College', 'University'])
    occupation = fake.job()
    residential_status = random.choice(['Owns house', 'Rents', 'Living with parents'])
    dependents = random.randint(0, 5)
    debt_to_income = round(random.uniform(0.1, 0.5), 2)
    credit_bureau = random.randint(760, 850)

    return {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Income Level': income_level,
        'Education': education,
        'Occupation': occupation,
        'Residential Status': residential_status,
        'Dependents': dependents,
        'Debt-to-Income': debt_to_income,
        'Credit_Bureau': credit_bureau
    }

def generate_inquiries(last_months):
    inquiries = []
    today = fake.date_this_month()

    for _ in range(random.randint(1, 5)):
        inquiry_date = fake.date_between(start_date=last_months, end_date=today)
        product_type = random.choice(['Personal Loan', 'Credit Card', 'Mortgage'])
        inquiries.append({'product_name': product_type, 'date': inquiry_date})

    return inquiries

def generate_dataset(num_rows, months):
    try:
        data_rows = []

        for _ in range(num_rows):
            customer_data = generate_customer_data()
            last_3_months_inquiries = generate_inquiries(months[0])
            last_6_months_inquiries = generate_inquiries(months[1])

            customer_row = {
                'Customer ID': fake.uuid4(),
                'Age': customer_data['Age'],
                'Gender': customer_data['Gender'],
                'Marital Status': customer_data['Marital Status'],
                'Income Level': customer_data['Income Level'],
                'Education': customer_data['Education'],
                'Occupation': customer_data['Occupation'],
                'Residential Status': customer_data['Residential Status'],
                'Dependents': customer_data['Dependents'],
                'Debt-to-Income': customer_data['Debt-to-Income'],
                'Credit_Bureau': customer_data['Credit_Bureau'],
                'Fixed_Deposits': random.choice(list(Config.FD_RATES.keys())),
                'Fixed_Deposits_Rate': Config.FD_RATES[random.choice(list(Config.FD_RATES.keys()))],
                'Fixed_Deposits_Senior_Rate': Config.FD_RATES[random.choice(list(Config.FD_RATES.keys()))] + Config.FD_SENIOR_CITIZEN_RATE_PREMIUM if customer_data['Age'] >= 60 else Config.FD_RATES[random.choice(list(Config.FD_RATES.keys()))]
            }


            for product_type in ['Personal Loan', 'Credit Card', 'Mortgage']:
                inq_in_last_3_months = any(inq['product_name'] == product_type for inq in last_3_months_inquiries)
                customer_row[f'last_3months_{product_type.replace(" ", "_").lower()}_inq'] = inq_in_last_3_months

            for product_type in ['Personal Loan', 'Credit Card', 'Mortgage']:
                inq_in_last_6_months = any(inq['product_name'] == product_type for inq in last_6_months_inquiries)
                customer_row[f'last_6months_{product_type.replace(" ", "_").lower()}_inq'] = inq_in_last_6_months

            data_rows.append(customer_row)
        return data_rows
    except Exception as e:
        st.error(f"🔴 Error generating dataset: {str(e)}")
        raise

    
@st.cache_data
def load_and_process_data(file_path, chunk_size=1000):
    try:
        logging.info(f"Loading data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for the file {file_path}")
        
        processed_data = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                processed_data.append(create_prompt_response(row))
        return processed_data
    
    except Exception as e:
        
        return []



def create_prompt_response(entry):
    prompt = (
        f"I'm a {entry['Age']}-year-old {entry['Gender']} looking to invest in {entry['Avenue']} "
        f"for {entry['Purpose']} over the next {entry['Duration']} years. What are my options?"
    )
    response = (
        f"Based on your preferences, here are your investment options:\n"
        f"- Fixed Deposits: {entry['Fixed_Deposits']} offers a rate of {entry['Fixed_Deposits_Rate']:.2f}% for regular customers and {entry['Fixed_Deposits_Senior_Rate']:.2f}% for senior citizens.\n"
        # f"Based on your preferences, here are your investment options:\n"
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
            algorithm="brute",
            n_neighbors=5
        )
        return vectordb
    except Exception as e:
        logging.error(f"An error occurred while creating the vector database: {e}")
        st.error(f"An error occurred while creating the vector database: {e}")
        return None

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


@st.cache_resource
def create_qa_chain(_sklearn_store):
    logging.info("Creating QA chain")
    openai_llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="Based on the following customer data and financial information: {context}, suggest suitable banking lending products and investment strategies in the following format:\n\n1. Product/Strategy 1: Description\n2. Product/Strategy 2: Description\n3. Product/Strategy 3: Description\nProvide detailed recommendations."
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=openai_llm,
        chain_type="stuff",
        retriever=_sklearn_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def get_fd_rates():
    return Config.FD_RATES

# User profile and risk assessment functions
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


# Investment advice and visualization functions
def get_investment_advice(profile, question, qa_chain):
    logging.info("Getting investment advice")
    prompt = f"I'm a {profile['age']}-year-old {profile['gender']} looking to invest in {profile['Avenue']} " \
             f"for {profile['Purpose']} over the next {profile['Duration']}. " \
             f"My risk assessment score is {profile['risk_score']}. {question}"
    response = qa_chain({"query": prompt})
    return response["result"]


def create_risk_profile_chart(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Profile", 'font': {'size': 24, 'color': '#E0E0E0'}},
        gauge={
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
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#E0E0E0", 'family': "Poppins"})
    return fig

def create_investment_allocation_chart(advice):
    allocations = {
        'Stocks': 40,
        'Bonds': 30,
        'Real Estate': 20,
        'Fixed Deposits': 10
    }
    
    fig = px.pie(
        values=list(allocations.values()),
        names=list(allocations.keys()),
        title="Recommended Investment Allocation",
        hole=0.3
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0", family="Poppins"),
    )
    fig.update_traces(marker=dict(colors=['#3B82F6', '#10B981', '#F59E0B', '#E0E0E0']))
    return fig



# New functions for the integrated model
@st.cache_resource
def setup_data_and_vectorstore():
    # Generate sample data
    months = [date.today() - relativedelta(months=+3), date.today() - relativedelta(months=+6)]
    dataset = generate_dataset(50, months)
    df = pd.DataFrame(dataset)
    df['content'] = [f"Based on the following customer data: {data}, suggest suitable banking lending products." for data in dataset]
    documents = [Document(page_content=row["content"], metadata={"class": row["Age"]}) for _, row in df.iterrows()]

    # Load and process the CSV data
    csv_data = load_and_process_data(Config.DATA_FILE)
    csv_documents = create_documents(csv_data)

    # Combine all documents and create the vector store
    all_documents = documents + csv_documents
    texts = split_documents(all_documents)
    return create_vector_db(texts)


# New functions for enhanced features

@st.cache_data
def fetch_stock_data(symbol, period="1y"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def predict_stock_price(data, days=30):
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    
    X = data[['Days']]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    future_dates = pd.date_range(start=data['Date'].max(), periods=days+1)[1:]
    future_days = (future_dates - data['Date'].min()).days.values.reshape(-1, 1)
    
    predictions = model.predict(future_days)
    
    return future_dates, predictions

def rebalance_portfolio(current_allocation, target_allocation):
    rebalancing_actions = {}
    for asset, current_pct in current_allocation.items():
        target_pct = target_allocation.get(asset, 0)
        difference = target_pct - current_pct
        if abs(difference) > 0.1:  # Only rebalance if difference is more than 0.1%
            action = "Buy" if difference > 0 else "Sell"
            rebalancing_actions[asset] = f"{action} {abs(difference):.2f}%"
    return rebalancing_actions


def setup_retrieval_qa(sklearn_store):
    try:
        openai_llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="Based on the following customer data and financial information: {context}, suggest suitable banking lending products and investment strategies in the following format:\n\n1. Product/Strategy 1: Description\n2. Product/Strategy 2: Description\n3. Product/Strategy 3: Description\nProvide detailed recommendations."
        )
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=openai_llm,
            chain_type="stuff",
            retriever=sklearn_store.as_retriever()
        )
        return retrieval_qa
    except Exception as e:
        st.error(f"Error setting up retrieval QA: {str(e)}")
        raise


# Main application function
def main():
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Home"
    
    # st.markdown('<div class="stHeader glow-effect"><h1 style="text-align: center;">MarketWealth Genius: Your AI Financial Advisor 💎</h1></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4,  tab6 = st.tabs(["🏠 Home", "👤 Profile & Risk", "💡 Investment Advice", "📊 Financial Dashboard",  "🎓 Financial Education Hub"])
    
    with tab1:
        st.markdown("## Welcome to MarketWealth Genius! 🚀")
        st.markdown("Your personal AI-powered financial advisor, here to guide you through your investment journey.")
        
        st.markdown("### 📘 How to Use MarketWealth Genius")
        for line in Config.HOW_TO_USE.split('\n'):
            if line.strip():
                st.markdown(f"- {line.strip()}")
        
        st.markdown("### 🌟 Key Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🤖 AI-Powered Advice")
            st.write("Get personalized investment recommendations based on your profile and market data.")
        with col2:
            st.markdown("#### 📊 Risk Assessment")
            st.write("Understand your risk tolerance and how it affects your investment strategy.")
        with col3:
            st.markdown("#### 🎓 Educational Resources")
            st.write("Access a wealth of information to improve your financial literacy.")
        
        st.markdown("### 🚀 Get Started")
        if st.button("Begin Your Investment Journey"):
            st.session_state.active_tab = "Profile & Risk"
            # st.experimental_rerun()
    
    with tab2:
        st.markdown("## 👤 User Profile and Risk Assessment")
        
        if 'profile' not in st.session_state:
            st.session_state.profile = {
                "age": "",
                "gender": "Male",
                "Avenue": "",
                "Purpose": "",
                "Duration": "",
                "risk_score": 0
            }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Personal Information")
            st.session_state.profile["age"] = st.number_input("Age", min_value=18, max_value=100, value=int(st.session_state.profile["age"]) if st.session_state.profile["age"] else 30)
            st.session_state.profile["gender"] = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.profile["gender"]))
            st.session_state.profile["Avenue"] = st.selectbox("Investment Avenue", ["Stocks", "Bonds", "Real Estate", "Mutual Funds", "ETFs", "Cryptocurrencies"], index=0 if not st.session_state.profile["Avenue"] else ["Stocks", "Bonds", "Real Estate", "Mutual Funds", "ETFs", "Cryptocurrencies"].index(st.session_state.profile["Avenue"]))
            st.session_state.profile["Purpose"] = st.selectbox("Investment Purpose", ["Retirement", "Short-term Goals", "Long-term Growth", "Income Generation", "Capital Preservation"], index=0 if not st.session_state.profile["Purpose"] else ["Retirement", "Short-term Goals", "Long-term Growth", "Income Generation", "Capital Preservation"].index(st.session_state.profile["Purpose"]))
            st.session_state.profile["Duration"] = st.slider("Investment Duration (years)", 1, 30, value=int(st.session_state.profile["Duration"]) if st.session_state.profile["Duration"] else 10)
        
        with col2:
            st.sidebar.markdown("### Risk Assessment")
            user_answers = []
            for i, question in enumerate(Config.RISK_ASSESSMENT_QUESTIONS, 1):
                answer = st.sidebar.select_slider(question, options=['1', '2', '3', '4', '5'], key=f'question_{i}')
                user_answers.append(answer)
            
            if st.sidebar.button("Calculate Risk Profile"):
                try:
                    risk_score = calculate_risk_score(user_answers)
                    st.session_state.profile["risk_score"] = risk_score
                    st.sidebar.success(f"Your risk score: {risk_score:.2f}")
                    st.sidebar.plotly_chart(create_risk_profile_chart(risk_score), use_container_width=True)
                except ValueError as e:
                    st.sidebar.error(str(e))

            st.sidebar.header("We value your feedback! 🌟")

            feedback = st.sidebar.text_area("📝 Leave Feedback", "Share your thoughts and suggestions here...")
            rating = st.sidebar.slider("📊 Rate Your Experience", 1, 5, 3)
            suggestions = st.sidebar.text_input("🗣️ Share Suggestions", "Any ideas for new features or improvements?")
            issue = st.sidebar.text_area("🧩 Report Issues", "Describe any problems you encountered...")

            if st.sidebar.button("Submit Feedback"):
                st.sidebar.success("Thank you for your feedback! 👍")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Profile"):
                save_user_profile(st.session_state.profile)
                st.success("Profile saved successfully!")
        with col2:
            if st.button("Load Profile"):
                loaded_profile = load_user_profile()
                if loaded_profile:
                    st.session_state.profile.update(loaded_profile)
                    st.success("Profile loaded successfully!")
                else:
                    st.error("No profile found.")


        st.markdown("## 🔄 Portfolio Rebalancing")
        
        st.markdown("### Current Portfolio Allocation")
        current_allocation = {}
        for asset in ['Stocks', 'Bonds', 'Real Estate', 'Cash']:
            current_allocation[asset] = st.number_input(f"Current {asset} allocation (%)", min_value=0, max_value=100, value=25, key=f"current_{asset}")
        
        st.markdown("### Target Portfolio Allocation")
        target_allocation = {}
        for asset in ['Stocks', 'Bonds', 'Real Estate', 'Cash']:
            target_allocation[asset] = st.number_input(f"Target {asset} allocation (%)", min_value=0, max_value=100, value=25, key=f"target_{asset}")
        
        if st.button("Rebalance Portfolio"):
            # Check if allocations sum to 100%
            if sum(current_allocation.values()) != 100 or sum(target_allocation.values()) != 100:
                st.error("Both current and target allocations must sum to 100%. Please adjust your inputs.")
            else:
                rebalancing_actions = rebalance_portfolio(current_allocation, target_allocation)
                st.markdown("### Rebalancing Actions")
                for asset, action in rebalancing_actions.items():
                    st.write(f"- {asset}: {action}")
                
                # Create a bar chart to visualize the rebalancing
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(current_allocation.keys()), y=list(current_allocation.values()), name='Current Allocation'))
                fig.add_trace(go.Bar(x=list(target_allocation.keys()), y=list(target_allocation.values()), name='Target Allocation'))
                fig.update_layout(title='Portfolio Allocation: Current vs Target', barmode='group', xaxis_title='Asset Class', yaxis_title='Allocation (%)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate the total portfolio value (assuming $100,000 for this example)
                total_portfolio_value = 100000
                st.markdown("### Rebalancing Transactions")
                st.markdown(f"Assuming a total portfolio value of ${total_portfolio_value:,}")
                for asset, action in rebalancing_actions.items():
                    action_type, percentage = action.split()
                    percentage = float(percentage.strip('%'))
                    amount = total_portfolio_value * (percentage / 100)
                    if action_type == "Buy":
                        st.write(f"- {asset}: Buy ${amount:,.2f}")
                    else:
                        st.write(f"- {asset}: Sell ${amount:,.2f}")
    



        with tab3:
            st.markdown("## 💡 Investment Advice")

            if 'qa_chain' not in st.session_state:
                st.session_state.qa_chain = None

            if st.button("Load Financial Data"):
                with st.spinner("Loading data..."):
                    sklearn_store = setup_data_and_vectorstore()
                    if sklearn_store:
                        st.session_state.qa_chain = create_qa_chain(sklearn_store)
                        st.success("Data loaded successfully. You can now ask for investment advice!")
                    else:
                        st.error("Failed to load data.")

            question = st.text_area("What would you like to know about investing?", height=100)

            if st.button("Get Personalized Advice"):
                if st.session_state.qa_chain:
                    with st.spinner("Generating your personalized investment advice..."):
                        response = st.session_state.qa_chain({"query": question})
                        st.markdown("### 🎯 Your Personalized Investment Advice")
                        st.info(response["result"])

                        # Add FD rate information
                        fd_rates = get_fd_rates()
                        st.markdown("#### Fixed Deposit Rates")
                        for bank, rate in fd_rates.items():
                            st.write(f"- {bank}: {rate:.2f}%")

                        st.plotly_chart(create_investment_allocation_chart(response["result"]), use_container_width=True)
                else:
                    st.error("Please load the financial data first by clicking the 'Load Financial Data' button.")


   


    
    with tab4:
        st.markdown("## 📊 Financial Dashboard")
        
        # Mock financial data
        savings = 15000
        investments = 50000
        debt = 5000
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Savings", f"${savings:,}", "+5%")
        
        with col2:
            st.metric("Investments", f"${investments:,}", "+12%")
        
        with col3:
            st.metric("Debt", f"${debt:,}", "-10%")
        
        # Interactive chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=['Jan', 'Feb', 'Mar', 'Apr', 'May'], y=[10000, 12000, 11000, 15000, 16000], name='Savings'))
        fig.add_trace(go.Scatter(x=['Jan', 'Feb', 'Mar', 'Apr', 'May'], y=[40000, 42000, 45000, 48000, 50000], name='Investments'))
        fig.update_layout(title='Financial Growth Over Time', xaxis_title='Month', yaxis_title='Amount ($)')
        st.plotly_chart(fig, use_container_width=True)

        # Stock prediction feature
        st.markdown("### 📈 Stock Price Prediction")
        stock_symbol = st.text_input("Enter a stock symbol (e.g., AAPL for Apple):")
        if stock_symbol:
            data = fetch_stock_data(stock_symbol)
            if data is not None:
                st.line_chart(data['Close'])
                future_dates, predictions = predict_stock_price(data)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical'))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Predicted'))
                fig.update_layout(title=f'{stock_symbol} Stock Price Prediction', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig, use_container_width=True)

 


    with tab6:
        

        st.markdown("## 🎓 Financial Education Hub")
        st.markdown("### 📚 Educational Resources")
        with st.expander("📘 Investment Basics"):
            st.write("Learn the fundamental concepts of investing, including asset classes, risk vs. return, and diversification.")
            st.markdown('[Learn more about Investment Basics](https://www.investopedia.com/terms/i/investment.asp)')
        
        with st.expander("🛡️ Risk Management Strategies"):
            st.write("Discover techniques to manage and mitigate investment risks, including portfolio diversification and hedging strategies.")
            st.markdown('[Learn more about Risk Management Strategies](https://www.sailpoint.com/identity-library/what-is-risk-management-strategy/)')
        
        with st.expander("💰 Tax-Efficient Investing"):
            st.write("Explore strategies to minimize your tax liability while maximizing your investment returns.")
            st.markdown('[Learn more about Tax-Efficient Investing](https://www.financialexpress.com/money/income-tax-tax-efficient-investment-planning-a-practical-guide-to-wealth-building-for-indian-investors-3384958/)')
        
        with st.expander("📅 Retirement Planning"):
            st.write("Learn how to plan and save effectively for your retirement, including information on 401(k)s, IRAs, and other retirement accounts.")
            st.markdown('[Learn more about Retirement Planning](https://www.investopedia.com/terms/r/retirement-planning.asp)')
        
        with st.expander("📈 Market Analysis Techniques"):
            st.write("Discover various methods for analyzing financial markets, including fundamental and technical analysis.")
            st.markdown('[Learn more about Market Analysis Techniques](https://nwokediothniel.medium.com/understanding-market-analysis-techniques-a-comprehensive-guide-928a124a2e7b)')

        
        st.markdown("### 🤔 Sample Questions to Get You Started")
        for category, questions in Config.SAMPLE_QUESTIONS.items():
            with st.expander(category):
                for q in questions:
                    st.write(f"• {q}")

        # Interactive Learning Modules
        st.markdown("### 🧠 Interactive Learning Modules")
        module = st.selectbox("Choose a learning module:", ["Investment Strategies", "Risk Assessment", "Financial Planning"])
        
        if module == "Investment Strategies":
            st.write("This module covers various investment strategies.")
            strategy = st.radio("Select a strategy to learn more:", ["Value Investing", "Growth Investing", "Index Investing"])
            if strategy:
                st.write(f"You selected {strategy}. Here's a brief overview...")
        elif module == "Risk Assessment":
            st.write("Learn how to assess your risk tolerance.")
            risk_score = st.slider("Rate your risk tolerance:", 1, 10, 5)
            st.write(f"Based on your score of {risk_score}, your risk tolerance is {'Low' if risk_score <= 3 else 'Medium' if risk_score <= 7 else 'High'}.")
        elif module == "Financial Planning":
            st.write("Create a basic financial plan.")
            income = st.number_input("Enter your monthly income:", min_value=0)
            expenses = st.number_input("Enter your monthly expenses:", min_value=0)
            if income and expenses:
                savings = income - expenses
                st.write(f"Your monthly savings potential is: ${savings}")

        # Financial Calculator
        st.markdown("### 🧮 Financial Calculator")
        calc_type = st.selectbox("Choose a calculator:", ["Compound Interest", "Loan Repayment", "Retirement Savings"])
        
        if calc_type == "Compound Interest":
            principal = st.number_input("Initial investment:", min_value=0.0)
            rate = st.number_input("Annual interest rate (%):", min_value=0.0, max_value=100.0)
            time = st.number_input("Time period (years):", min_value=0)
            if st.button("Calculate Compound Interest"):
                result = principal * (1 + rate/100)**time
                st.write(f"After {time} years, your investment will grow to: ${result:.2f}")
        elif calc_type == "Loan Repayment":
            loan_amount = st.number_input("Loan amount:", min_value=0.0)
            loan_term = st.number_input("Loan term (years):", min_value=0)
            interest_rate = st.number_input("Annual interest rate (%):", min_value=0.0, max_value=100.0)
            if st.button("Calculate Monthly Payment"):
                monthly_rate = interest_rate / (12 * 100)
                months = loan_term * 12
                payment = loan_amount * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
                st.write(f"Your monthly payment will be: ${payment:.2f}")
        elif calc_type == "Retirement Savings":
            current_age = st.number_input("Current age:", min_value=0, max_value=100)
            retirement_age = st.number_input("Retirement age:", min_value=current_age, max_value=100)
            monthly_contribution = st.number_input("Monthly contribution:", min_value=0.0)
            annual_return = st.number_input("Expected annual return (%):", min_value=0.0, max_value=100.0)
            if st.button("Calculate Retirement Savings"):
                months = (retirement_age - current_age) * 12
                total_savings = monthly_contribution * ((1 + annual_return/1200)**(months) - 1) / (annual_return/1200)
                st.write(f"At retirement, your savings could grow to: ${total_savings:.2f}")

        # Glossary of Financial Terms
        st.markdown("### 📖 Glossary of Financial Terms")
        terms = {
            "Asset Allocation": "The process of dividing investments among different kinds of assets, such as stocks, bonds, and cash, to optimize the risk/reward tradeoff based on an individual's specific situation and goals.",
            "Diversification": "A risk management strategy that mixes a wide variety of investments within a portfolio to potentially limit exposure to any single asset or risk.",
            "Compound Interest": "Interest calculated on the initial principal and also on the accumulated interest of previous periods of a deposit or loan.",
            "Dollar-Cost Averaging": "An investment strategy in which an investor divides up the total amount to be invested across periodic purchases of a target asset in an effort to reduce the impact of volatility on the overall purchase.",
            "Exchange-Traded Fund (ETF)": "A type of security that tracks an index, sector, commodity, or other asset, but which can be purchased or sold on a stock exchange the same as a regular stock."
        }
        term = st.selectbox("Select a term to learn more:", list(terms.keys()))
        st.write(terms[term])

        # Financial News Feed
        st.markdown("### 📰 Latest Financial News")
        # In a real application, you would fetch this data from a financial news API
        news_items = [
            "Stock Market Reaches New High",
            "Federal Reserve Announces Interest Rate Decision",
            "Tech Stocks Rally on Positive Earnings Reports",
            "Global Economic Outlook Improves",
            "New Cryptocurrency Regulations Proposed"
        ]
        for item in news_items:
            st.write(f"• {item}")

        st.markdown("---")
        st.markdown('<p style="text-align: center;">Created with ❤ using Streamlit and LangChain</p>', unsafe_allow_html=True)

        # Chatbot interface
        st.markdown("### 💬 Chat with MarketWealth Genius")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything about investing!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if st.session_state.qa_chain:
                    message_placeholder = st.empty()
                    full_response = ""
                    for response in st.session_state.qa_chain.run(prompt):
                        full_response += response
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    st.markdown("Please load the financial data first by clicking the 'Load Financial Data' button in the Investment Advice tab.")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})







if __name__ == "__main__":
    logging.info("Starting MarketWealth Genius: Your AI Financial Advisor")
    main()

