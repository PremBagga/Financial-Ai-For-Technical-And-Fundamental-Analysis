import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from phi.agent.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
import yfinance as yf
# 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
# Get API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Enhanced stock symbol mappings
COMMON_STOCKS = {
    # US Stocks
    'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'GOOGLE': 'GOOGL', 'MICROSOFT': 'MSFT',
    'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'META': 'META', 'NETFLIX': 'NFLX',
    'ADOBE': 'ADBE', 'INTEL': 'INTC', 'AMD': 'AMD', 'QUALCOMM': 'QCOM',
    'PAYPAL': 'PYPL', 'VISA': 'V', 'MASTERCARD': 'MA', 'JOHNSON & JOHNSON': 'JNJ',
    'WALMART': 'WMT', 'CISCO': 'CSCO', 'PEPSICO': 'PEP', 'COCA-COLA': 'KO',
    'MCDONALDS': 'MCD', 'DISNEY': 'DIS', 'BOEING': 'BA', 'FORD': 'F',
    'GENERAL MOTORS': 'GM', 'STARBUCKS': 'SBUX',
    # Indian Stocks - NSE
    'TCS': 'TCS.NS', 'RELIANCE': 'RELIANCE.NS', 'INFOSYS': 'INFY.NS', 'WIPRO': 'WIPRO.NS',
    'HDFC': 'HDFCBANK.NS', 'TATAMOTORS': 'TATAMOTORS.NS', 'ICICIBANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS', 'MARUTI': 'MARUTI.NS', 'BHARTIARTL': 'BHARTIARTL.NS',
    'HCLTECH': 'HCLTECH.NS', 'ITC': 'ITC.NS', 'AXISBANK': 'AXISBANK.NS',
    'LT': 'LT.NS', 'SUNPHARMA': 'SUNPHARMA.NS', 'BAJFINANCE': 'BAJFINANCE.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS', 'HINDUNILVR': 'HINDUNILVR.NS', 'ASIANPAINT': 'ASIANPAINT.NS',
    'KOTAKBANK': 'KOTAKBANK.NS', 'TITAN': 'TITAN.NS', 'TECHM': 'TECHM.NS',
    'GRASIM': 'GRASIM.NS', 'HINDALCO': 'HINDALCO.NS', 'ONGC': 'ONGC.NS'
}

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stock-header {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        border-radius: 10px;
    }
    .news-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateX(5px);
    }
    .stButton>button {
        width: 100%;
    }
    .market-indicator {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.watchlist = set()
    st.session_state.analysis_history = []
    st.session_state.last_refresh = None

def initialize_agents():
    """Initialize all agent instances with improved error handling"""
    if not st.session_state.agents_initialized:
        try:
            st.session_state.web_agent = Agent(
                name="Web Search Agent",
                role="Search the web for the information",
                model=Groq(api_key=GROQ_API_KEY),
                tools=[
                    GoogleSearch(fixed_language='english', fixed_max_results=5),
                    DuckDuckGo(fixed_max_results=5)
                ],
                instructions=['Always include sources and verification'],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.finance_agent = Agent(
                name="Financial AI Agent",
                role="Providing financial insights",
                model=Groq(api_key=GROQ_API_KEY),
                tools=[
                    YFinanceTools(
                        stock_price=True,
                        company_news=True,
                        analyst_recommendations=True,
                        historical_prices=True
                    )
                ],
                instructions=["Provide detailed analysis with data visualization"],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.multi_ai_agent = Agent(
                name='A Stock Market Agent',
                role='A comprehensive assistant specializing in stock market analysis',
                model=Groq(api_key=GROQ_API_KEY),
                team=[st.session_state.web_agent, st.session_state.finance_agent],
                instructions=["Provide comprehensive analysis with multiple data sources"],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.agents_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing agents: {str(e)}")
            return False

def get_symbol_from_name(stock_name):
    """Enhanced function to fetch stock symbol from full stock name"""
    try:
        # Clean up input
        stock_name = stock_name.strip().upper()
        
        # First check if it's in our common stocks dictionary
        if stock_name in COMMON_STOCKS:
            return COMMON_STOCKS[stock_name]
        
        # Check if it's already a valid symbol
        ticker = yf.Ticker(stock_name)
        try:
            info = ticker.info
            if info and 'symbol' in info:
                return stock_name
        except:
            pass
        
        # Try Indian stock market (NSE)
        try:
            indian_symbol = f"{stock_name}.NS"
            ticker = yf.Ticker(indian_symbol)
            info = ticker.info
            if info and 'symbol' in info:
                return indian_symbol
        except:
            # Try BSE
            try:
                bse_symbol = f"{stock_name}.BO"
                ticker = yf.Ticker(bse_symbol)
                info = ticker.info
                if info and 'symbol' in info:
                    return bse_symbol
            except:
                pass
        
        st.error(f"Could not find valid symbol for {stock_name}")
        return None
    except Exception as e:
        st.error(f"Error processing {stock_name}: {str(e)}")
        return None

def get_stock_data(symbol, period="1y"):
    """Enhanced function to fetch stock data with proper cache handling"""
    try:
        # Create a new ticker instance
        stock = yf.Ticker(symbol)
        
        # Fetch data with error handling
        try:
            info = stock.info
            if not info:
                raise ValueError("No data retrieved for symbol")
        except Exception as info_error:
            # If .NS suffix is missing for Indian stocks, try adding it
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                try:
                    indian_symbol = f"{symbol}.NS"
                    stock = yf.Ticker(indian_symbol)
                    info = stock.info
                    symbol = indian_symbol
                except:
                    # Try Bombay Stock Exchange
                    try:
                        bse_symbol = f"{symbol}.BO"
                        stock = yf.Ticker(bse_symbol)
                        info = stock.info
                        symbol = bse_symbol
                    except:
                        raise info_error
            else:
                raise info_error

        # Fetch historical data
        hist = stock.history(period=period, interval="1d", auto_adjust=True)
        
        if hist.empty:
            raise ValueError("No historical data available")
            
        return info, hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def create_price_chart(hist_data, symbol):
    """Create an interactive price chart using plotly"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='Price'
    ))
    
        st.markdown(f"<div class='market-indicator'>Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</div>", 
                   unsafe_allow_html=True)

if __name__ == "__main__":
    main() 

#venv/scripts/activate
#python -m streamlit run financial_agent.py    
