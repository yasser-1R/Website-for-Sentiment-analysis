import streamlit as st
import joblib
import os
import sys
import numpy as np

# Add the current directory to the path so Python can find the preprocessing module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pre-processing functions from preprocessing.py
from preprocessing import arabic_preprocessing, english_preprocessing

# Helper function for sigmoid transform (used for SVC decision scores)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ========== Mapping models ==========
model_mappings = {
    "Arabic Tweets": {  # Changed from ASTD
        "folder": "ASTD_Models",
        "vectorizer": "tfidf_vectorizer_ASTD_Pre_Blc.pkl",
        "preprocessing": arabic_preprocessing,
        "language": "Arabic",
        "models": {
            "MultinomialNB": "MultinomialNB_ASTD_Preprocessed_Balanced_best.pkl",
            "Linear SVC": "LinearSVC_ASTD_Preprocessed_Balanced_best.pkl",
            "Logistic Regression": "LogisticRegression_ASTD_Preprocessed_Balanced_best.pkl",
            "Random Forest": "RandomForestClassifier_ASTD_Preprocessed_Balanced_best.pkl",
            "K-Nearest Neighbors": "KNeighborsClassifier_ASTD_Preprocessed_Balanced_best.pkl",
            "AdaBoost": "AdaBoostClassifier_ASTD_Preprocessed_Balanced_best.pkl"
        }
    },
    "Arabic Reviews": {  # Changed from LABR
        "folder": "LABR_Models",
        "vectorizer": "tfidf_vectorizer_LABR_Pre_Blc.pkl",
        "preprocessing": arabic_preprocessing,
        "language": "Arabic",
        "models": {
            "MultinomialNB": "MultinomialNB_LABR_Preprocessed_Balanced_best.pkl",
            "Linear SVC": "LinearSVC_LABR_Preprocessed_Balanced_best.pkl",
            "Logistic Regression": "LogisticRegression_LABR_Preprocessed_Balanced_best.pkl",
            "Random Forest": "RandomForestClassifier_LABR_Preprocessed_Balanced_best.pkl",
            "K-Nearest Neighbors": "KNeighborsClassifier_LABR_Preprocessed_Balanced_best.pkl",
            "AdaBoost": "AdaBoostClassifier_LABR_Preprocessed_Balanced_best.pkl"
        }
    },
    "English Reviews": {  # Changed from IMDB
        "folder": "IMDB_Models",
        "vectorizer": "tfidf_vectorizer_IMDB_Pre.pkl",
        "preprocessing": english_preprocessing,
        "language": "English",
        "models": {
            "MultinomialNB": "MultinomialNB_IMDB_Preprocessed_best.pkl",
            "Linear SVC": "LinearSVC_IMDB_Preprocessed_best.pkl",
            "Logistic Regression": "LogisticRegression_IMDB_Preprocessed_best.pkl",
            "Random Forest": "RandomForestClassifier_IMDB_Preprocessed_best.pkl",
            "K-Nearest Neighbors": "KNeighborsClassifier_IMDB_Preprocessed_best.pkl",
            "AdaBoost": "AdaBoostClassifier_IMDB_Preprocessed_best.pkl"
        }
    },
    "English Tweets": {  # Changed from TESA
        "folder": "TESA_Models",
        "vectorizer": "tfidf_vectorizer_TESA_Pre.pkl",
        "preprocessing": english_preprocessing,
        "language": "English",
        "models": {
            "MultinomialNB": "MultinomialNB_TESA_Preprocessed_best.pkl",
            "Linear SVC": "LinearSVC_TESA_Preprocessed_best.pkl",
            "Logistic Regression": "LogisticRegression_TESA_Preprocessed_best.pkl",
            "Random Forest": "RandomForestClassifier_TESA_Preprocessed_best.pkl",
            "K-Nearest Neighbors": "KNeighborsClassifier_TESA_Preprocessed_best.pkl",
            "AdaBoost": "AdaBoostClassifier_TESA_Preprocessed_best.pkl"
        }
    }
}

# Arabic translations
arabic_translations = {
    "Select Dataset": "اختر مجموعة البيانات",
    "Select Model Algorithm": "اختر خوارزمية النموذج",
    "Enter text for sentiment analysis:": "أدخل النص للتحليل العاطفي:",
    "Type or paste text here...": "اكتب أو الصق النص هنا...",
    "Analyze Sentiment": "تحليل المشاعر",
    "Please enter some text.": "الرجاء إدخال بعض النص.",
    "Processing text...": "معالجة النص...",
    "Preprocessed Text:": "النص المعالج:",
    "Predicted Sentiment:": "المشاعر المتوقعة:",
    "Positive": "إيجابي",
    "Negative": "سلبي",
    "Confidence": "الثقة",
    "Sentiment Distribution": "توزيع المشاعر",
    "Negative Probability": "احتمالية السلبية",
    "Positive Probability": "احتمالية الإيجابية",
    "Sentiment Analysis": "تحليل المشاعر",
    "Preprocessing resulted in empty text.": "أدت المعالجة المسبقة إلى نص فارغ."
}

# Function to translate based on language
def translate(text, language):
    if language == "Arabic":
        return arabic_translations.get(text, text)
    return text

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# ========== Sidebar: Dataset & Model selection ==========
# Get or initialize the selected dataset in session state
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = list(model_mappings.keys())[0]

# Determine language based on selected dataset
current_dataset = st.sidebar.selectbox(
    "Select Dataset",
    list(model_mappings.keys()),
    key="dataset_selector"
)

# Update current language when dataset changes
current_language = model_mappings[current_dataset]["language"]

# Sidebar title with translation
st.sidebar.title(translate("Sentiment Analysis", current_language))

# Model selection based on dataset
selected_model_name = st.sidebar.selectbox(
    translate("Select Model Algorithm", current_language),
    list(model_mappings[current_dataset]["models"].keys())
)

# Update session state
st.session_state.selected_dataset = current_dataset

# ========== Apply new color theme CSS ==========
st.markdown("""
    <style>
    /* New color theme */
    .stApp {
        background-color: #213448;
        color: #ECEFCA;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1v3fvcr {
        background-color: #213448;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #94B4C1 !important;
    }
    
    /* Text area and input fields */
    .stTextArea textarea, .stTextInput input {
        background-color: #547792;
        color: #ECEFCA;
        border: 1px solid #94B4C1;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #547792;
        color: #ECEFCA;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #94B4C1;
        color: #213448;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #94B4C1 !important;
    }
    
    /* Code blocks */
    .stCode {
        background-color: #547792;
        border: 1px solid #94B4C1;
        color: #ECEFCA;
    }
    
    /* Prediction result boxes */
    .prediction-positive {
        background-color: #547792;
        color: #ECEFCA;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        margin-top: 15px;
        border-left: 4px solid #94B4C1;
    }
    
    .prediction-negative {
        background-color: #213448;
        color: #ECEFCA;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        margin-top: 15px;
        border-left: 4px solid #547792;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #547792;
        color: #ECEFCA;
    }
    
    /* Hover effects for interactive elements */
    .stSelectbox:hover > div {
        border-color: #94B4C1;
    }
    
    /* Small text */
    small {
        color: #ECEFCA;
    }
    </style>
    """, unsafe_allow_html=True)

# ========== Load the correct vectorizer and model ==========
try:
    vectorizer_path = os.path.join(model_mappings[current_dataset]["folder"], model_mappings[current_dataset]["vectorizer"])
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        st.error(f"Vectorizer file not found: {vectorizer_path}")
        st.stop()
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")
    st.stop()

try:
    model_path = os.path.join(model_mappings[current_dataset]["folder"], model_mappings[current_dataset]["models"][selected_model_name])
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error(f"Model file not found: {model_path}")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ========== Main Page ==========
st.title(translate("Sentiment Analysis", current_language))

# ========== Text Input ==========
user_input = st.text_area(
    translate("Enter text for sentiment analysis:", current_language),
    height=150,
    placeholder=translate("Type or paste text here...", current_language)
)

# ========== Predict ==========
if st.button(translate("Analyze Sentiment", current_language), key="analyze_button", use_container_width=True):
    if user_input.strip() == "":
        st.warning(translate("Please enter some text.", current_language))
    else:
        try:
            # Get the appropriate preprocessing function
            preprocess_fn = model_mappings[current_dataset]["preprocessing"]
            
            with st.spinner(translate("Processing text...", current_language)):
                # Preprocessing
                preprocessed_text = preprocess_fn(user_input)

                # Display preprocessed text directly
                st.subheader(translate("Preprocessed Text:", current_language))
                st.code(preprocessed_text, language='text')

                # Display warning if preprocessing resulted in empty text
                if not preprocessed_text.strip():
                    st.warning(translate("Preprocessing resulted in empty text.", current_language))
                    st.stop()

                # Prediction
                X_input = vectorizer.transform([preprocessed_text])
                prediction = model.predict(X_input)[0]

                # Get prediction probabilities
                prediction_proba = None
                confidence = None
                
                # Handle different model types for probability extraction
                if hasattr(model, "predict_proba"):
                    # For models with direct probability output (MultinomialNB, RandomForest, etc.)
                    prediction_proba = model.predict_proba(X_input)[0]
                    confidence = prediction_proba.max() * 100
                elif hasattr(model, "decision_function"):
                    # For models with decision function (SVC, etc.)
                    decision_scores = model.decision_function(X_input)[0]
                    # Convert to probability-like score between 0 and 1
                    if isinstance(decision_scores, np.ndarray):
                        # Multi-class case
                        scores = decision_scores
                    else:
                        # Binary case
                        scores = np.array([1 - sigmoid(decision_scores), sigmoid(decision_scores)])
                    confidence = scores.max() * 100
                else:
                    # Fallback for models without probability estimates
                    confidence = None

                # Display Prediction
                label_mapping = {0: translate("Negative", current_language), 1: translate("Positive", current_language)}
                prediction_label = label_mapping.get(prediction, "Unknown")
                
                # Display prediction with styling based on sentiment
                if prediction_label == translate("Positive", current_language):
                    st.markdown(f"""
                    <div class="prediction-positive">
                        <p>😊 {translate("Predicted Sentiment:", current_language)} {prediction_label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-negative">
                        <p>😞 {translate("Predicted Sentiment:", current_language)} {prediction_label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show confidence percentage if available
                if confidence is not None:
                    # Display confidence with progress bar
                    st.markdown(f"<small>**{translate('Confidence', current_language)}**: {confidence:.2f}%</small>", unsafe_allow_html=True)
                    st.progress(confidence/100)
                
                # Show detailed class probabilities if available
                if prediction_proba is not None and len(prediction_proba) == 2:
                    st.markdown(f"<small>**{translate('Sentiment Distribution', current_language)}**</small>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<small>{translate('Negative Probability', current_language)}: {prediction_proba[0]*100:.2f}%</small>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<small>{translate('Positive Probability', current_language)}: {prediction_proba[1]*100:.2f}%</small>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc(), language="python")