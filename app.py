import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Cinemix Revenue Predictor",
    page_icon="🎬",
    layout="wide"
)

# --- Custom Styling (Premium Dark Mode with Glassmorphism) ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: radial-gradient(circle at top right, #0d1117, #161b22);
        color: #ffffff;
    }
    
    /* Content Card Container */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background: rgba(30, 39, 50, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 20px !important;
    }
    
    /* Labels and normal text */
    label, p, span, div {
        color: #c9d1d9 !important;
    }

    /* Input Fields Styling - Premium Visibility */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stDateInput input {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 10px !important;
        transition: all 0.2s;
    }
    
    /* Dropdown text visibility */
    div[data-baseweb="select"] > div {
        color: white !important;
        background-color: #1e293b !important;
    }
    
    /* Dropdown icon visibility */
    svg[title="open"] {
        fill: white !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }

    /* Placeholder text styling */
    ::placeholder {
        color: #9ca3af !important;
        opacity: 0.8;
    }

    /* Center Info/Helper Text */
    div[data-testid="stMarkdownContainer"] p {
        text-align: center;
    }
    
    /* Specific adjustment for labels */
    label[data-testid="stWidgetLabel"] p {
        text-align: left !important;
        font-weight: 600;
        margin-bottom: 10px;
        color: #58a6ff !important;
    }

    /* Result Card - High Contrast */
    .result-card {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 12px 32px rgba(0,0,0,0.6);
        margin: 30px 0;
        animation: fadeIn 0.8s ease-out;
    }
    
    .result-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .result-label {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Metric styles */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-weight: 600;
    }
    
    /* Metric delta (change indicator) styling */
    [data-testid="stMetricDelta"] {
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    
    /* Positive delta (green) */
    [role="img"][aria-label*="increase"], 
    .metric-positive {
        color: #3fb950 !important;
    }
    
    /* Negative delta (red) */
    [role="img"][aria-label*="decrease"],
    .metric-negative {
        color: #f85149 !important;
    }

    /* Form Inputs */
    .stNumberInput, .stSlider, .stSelectbox {
        margin-bottom: 25px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #f72585, #7209b7);
        color: white;
        border: none;
        padding: 14px 20px !important;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 10px !important;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 10px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(247, 37, 133, 0.5);
    }

    /* Footer Styling - Centered and Spaced */
    .footer {
        margin-top: 120px;
        padding: 50px 0;
        text-align: center;
        width: 100%;
        color: #8b949e;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- Global Constants (Based on Model Training) ---
# Updated to match the actual top features from the trained model
TOP_GENRES = ['Drama', 'Comedy', 'Action', 'Thriller', 'Adventure', 'Romance', 'Horror', 'Crime', 'Fantasy', 'Science Fiction']
TOP_LANGS = ['en', 'es', 'fr', 'ja', 'ko']

# Language mapping for display
LANG_MAP = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Other': 'other'
}

# --- Helper Functions ---
@st.cache_resource
def load_model():
    return joblib.load('movie_revenue_file.pkl')

def prepare_features(inputs, feature_names):
    # Reconstruct the feature vector in exact order expected by the model
    data = {}
    
    # Map full language name back to code
    lang_code = LANG_MAP.get(inputs['lang'], 'other')
    
    # Numerical base features
    data['vote_average'] = inputs['vote_average']
    data['vote_count'] = inputs['vote_count']
    data['runtime'] = inputs['runtime']
    data['popularity'] = inputs['popularity']
    data['release_year'] = inputs['release_year']
    data['release_month'] = inputs['release_month']
    
    # Handling Genres dynamically based on expected feature names
    for col in feature_names:
        if col.startswith('genre_'):
            genre_name = col.replace('genre_', '')
            data[col] = 1 if genre_name in inputs['genres'] else 0
            
    # Handling Languages dynamically
    for col in feature_names:
        if col.startswith('lang_'):
            col_lang_code = col.replace('lang_', '')
            if col_lang_code == 'other':
                # 'known_langs' are those that have their own 'lang_xx' column
                known_langs = [c.replace('lang_', '') for c in feature_names if c.startswith('lang_') and c != 'lang_other']
                data[col] = 1 if lang_code not in known_langs else 0
            else:
                data[col] = 1 if lang_code == col_lang_code else 0
    
    # Log Budget
    data['log_budget'] = np.log1p(inputs['budget'])
    
    # Final check and ordering
    df = pd.DataFrame([data])
    return df[feature_names]

# --- UI Header ---
# Using a container and specific column ratios for centered-feeling layout
header_container = st.container()
with header_container:
    h_col1, h_col2 = st.columns([1, 4])
    
    with h_col1:
        # Align logo to the right of its column to be closer to title
        try:
            st.image('cinemix_logo.png', height=70)
        except:
            # Fallback if image not found
            st.markdown("### 🎬")

    with h_col2:
        # Vertical alignment via margin-top if needed, or just let them sit together
        st.markdown("<h1 style='margin-bottom: 0px;'>CINEMIX: MOVIE REVENUE PREDICTOR</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-style: italic; color: #8b949e; margin-top: -5px;'>Predict box office success using institutional grade machine learning models.</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Application Layout ---
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("🎥 Film Specifications")
    
    movie_title = st.text_input("Movie Title", placeholder="e.g., The Dark Knight")
    budget = st.number_input("Budget (USD)", min_value=0, value=50000000, step=1000000, help="Total production budget in dollars")
    
    c1, c2 = st.columns(2)
    with c1:
        runtime = st.slider("Runtime (min)", 0, 300, 110)
        popularity = st.slider("Popularity Index", 0.0, 1000.0)
    with c2:
        vote_avg = st.slider("Vote Average", 0.0, 10.0, 6.5)
        vote_count = st.number_input("Vote Count", min_value=0, value=1000)

    release_date = st.date_input("Planned Release Date")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎭 Categorical Metadata")
    
    genres = st.multiselect("Genres", TOP_GENRES, default=['Action', 'Adventure'])
    language = st.selectbox("Original Language", list(LANG_MAP.keys()), index=0)

with right_col:
    st.subheader("📈 Revenue Forecast")
    st.info("Fill out the specs on the left and hit the button below to generate a prediction.")
    
    predict_btn = st.button("Calculate Estimated Revenue")
    
    if predict_btn:
        try:
            model = load_model()
            
            # Get expected feature order
            feature_names = model.feature_names_in_.tolist()
            
            inputs = {
                'budget': budget,
                'runtime': runtime,
                'popularity': popularity,
                'vote_average': vote_avg,
                'vote_count': vote_count,
                'release_year': release_date.year,
                'release_month': release_date.month,
                'genres': genres,
                'lang': language
            }
            
            # Show spinner
            with st.spinner('Analyzing market trends and generating prediction...'):
                time.sleep(1.5) # Simulate processing for premium feel
                X = prepare_features(inputs, feature_names)
                log_pred = model.predict(X)[0]
                actual_pred = np.expm1(log_pred)
                
            # Aesthetic Result Card
            title_display = f" for '{movie_title}'" if movie_title else ""
            revenue_indicator = "+" if actual_pred > 0 else ""
            indicator_color = "#3fb950" if actual_pred > 0 else "#f85149"
            
            st.markdown(f"""
                <div class="result-card">
                    <p class="result-label">Estimated Box Office Revenue{title_display}</p>
                    <p class="result-value"><span style="color: {indicator_color};">{revenue_indicator}</span>${actual_pred:,.2f}</p>
                    <p style="color: rgba(255,255,255,0.7)">Based on similar historical data and market trends</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Contextual metrics
            m1, m2 = st.columns(2)

            with m1:
                roi = ((actual_pred - budget) / budget) * 100 if budget > 0 else 0
                roi_delta = roi if roi != 0 else None
                st.metric("Estimated ROI", f"{'+' if roi > 0 else ''}{roi:.1f}%", delta=roi_delta, delta_color="inverse")

            with m2:
                if roi < 0:
                    status = "Flop"
                elif roi < 50:
                    status = "Average"
                elif roi < 100:
                    status = "Hit"
                else:
                    status = "Blockbuster"

                st.metric("Success Tier", status)
                
        except FileNotFoundError:
            st.error("Error: 'movie_revenue_model.pkl' not found. Please ensure the model is trained and saved in the current directory.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>🎬 <b>Cinemix Intelligence</b> | Predictive Analytics for Global Cinema</p>
    <p style="font-size: 0.8rem; opacity: 0.6;">Powered by Scikit-Learn RandomForestRegressor & Streamlit</p>
</div>
""", unsafe_allow_html=True)
