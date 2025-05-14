import streamlit as st
import joblib
import os
import sys
import numpy as np
import pandas as pd
import io
import base64

# Add the current directory to the path so Python can find the preprocessing module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pre-processing functions from preprocessing.py
from preprocessing import arabic_preprocessing, english_preprocessing

# Helper function for sigmoid transform (used for SVC decision scores)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Helper function to download dataframe as CSV
def get_csv_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

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
    "Preprocessing resulted in empty text.": "أدت المعالجة المسبقة إلى نص فارغ.",
    "Single Text": "نص واحد",
    "Batch Processing": "معالجة الدفعة",
    "Upload CSV File": "تحميل ملف CSV",
    "Please upload a CSV file containing a 'Text' column.": "يرجى تحميل ملف CSV يحتوي على عمود 'Text'.",
    "Processing file...": "معالجة الملف...",
    "File processed successfully!": "تمت معالجة الملف بنجاح!",
    "Download Results": "تنزيل النتائج",
    "Results Summary": "ملخص النتائج",
    "Total Texts": "إجمالي النصوص",
    "Positive Texts": "النصوص الإيجابية",
    "Negative Texts": "النصوص السلبية",
    "Positive Percentage": "النسبة المئوية الإيجابية",
    "Negative Percentage": "النسبة المئوية السلبية",
    "Error processing file": "خطأ في معالجة الملف",
    "No valid text column found in the CSV": "لم يتم العثور على عمود نص صالح في ملف CSV",
    "Preview of Processed Data": "معاينة البيانات المعالجة"
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
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #547792;
        color: #ECEFCA;
        border: 1px dashed #94B4C1;
        padding: 15px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #547792;
        color: #ECEFCA;
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        border: 1px solid #94B4C1;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #94B4C1;
        color: #213448;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #547792;
        border-radius: 8px;
        padding: 10px;
        border-left: 4px solid #94B4C1;
    }
    
    /* Download link styling */
    a {
        color: #94B4C1;
        text-decoration: none;
        padding: 8px 12px;
        background-color: #547792;
        border-radius: 4px;
        display: inline-block;
        margin-top: 10px;
    }
    
    a:hover {
        background-color: #94B4C1;
        color: #213448;
    }
    
    /* Table styling */
    .stDataFrame {
        background-color: #547792;
        border-radius: 8px;
        padding: 10px;
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

# Create tabs for single text processing and batch processing
tab1, tab2 = st.tabs([
    translate("Single Text", current_language), 
    translate("Batch Processing", current_language)
])

# ========== Single Text Analysis Tab ==========
with tab1:
    # Text Input
    user_input = st.text_area(
        translate("Enter text for sentiment analysis:", current_language),
        height=150,
        placeholder=translate("Type or paste text here...", current_language)
    )

    # Predict
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

# ========== Batch Processing Tab ==========
with tab2:
    st.subheader(translate("Upload CSV File", current_language))
    
    # File uploader
    uploaded_file = st.file_uploader(
        translate("Please upload a CSV file containing a 'Text' column.", current_language),
        type=["csv"]
    )
    
    if uploaded_file is not None:
        try:
            # Try reading the CSV file with different encodings
            encodings_to_try = ['utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    # Reset file pointer to beginning before each read attempt
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"File successfully read using {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("Failed to read the CSV file with any of the supported encodings. Please check your file encoding and try again.")
                st.stop()
            
            # Check if "Text" or "text" column exists
            text_col = None
            for col in df.columns:
                if col.lower() == "text":
                    text_col = col
                    break
            
            if text_col is None:
                st.error(translate("No valid text column found in the CSV", current_language))
                st.stop()
            
            # Process button
            if st.button(translate("Analyze Sentiment", current_language), key="process_file_button", use_container_width=True):
                try:
                    with st.spinner(translate("Processing file...", current_language)):
                        # Get preprocessing function
                        preprocess_fn = model_mappings[current_dataset]["preprocessing"]
                        
                        # Create a copy of the dataframe with only the text column
                        result_df = df[[text_col]].copy()
                        
                        # Process each text and store results
                        preprocessed_texts = []
                        predictions = []
                        confidences = []
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        total_rows = len(df)
                        
                        for idx, text in enumerate(df[text_col]):
                            # Update progress bar
                            progress_bar.progress((idx + 1) / total_rows)
                            
                            if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                                # Handle empty, NaN, or non-string texts
                                preprocessed_texts.append("")
                                predictions.append(None)
                                confidences.append(None)
                                continue
                                
                            # Preprocess text
                            try:
                                preprocessed_text = preprocess_fn(str(text))
                                preprocessed_texts.append(preprocessed_text)
                                
                                if not preprocessed_text.strip():
                                    predictions.append(None)
                                    confidences.append(None)
                                    continue
                                
                                # Make prediction
                                X_input = vectorizer.transform([preprocessed_text])
                                prediction = model.predict(X_input)[0]
                                predictions.append(prediction)
                                
                                # Get confidence
                                if hasattr(model, "predict_proba"):
                                    prediction_proba = model.predict_proba(X_input)[0]
                                    confidence = prediction_proba.max() * 100
                                elif hasattr(model, "decision_function"):
                                    decision_scores = model.decision_function(X_input)[0]
                                    if isinstance(decision_scores, np.ndarray):
                                        scores = decision_scores
                                    else:
                                        scores = np.array([1 - sigmoid(decision_scores), sigmoid(decision_scores)])
                                    confidence = scores.max() * 100
                                else:
                                    confidence = None
                                
                                confidences.append(confidence)
                                
                            except Exception as e:
                                # Handle errors for individual rows instead of failing the entire process
                                st.warning(f"Error processing row {idx+1}: {str(e)}")
                                preprocessed_texts.append("ERROR")
                                predictions.append(None)
                                confidences.append(None)
                        
                        # Convert numeric labels to text labels
                        label_mapping = {
                            0: translate("Negative", current_language), 
                            1: translate("Positive", current_language),
                            None: "N/A"
                        }
                        
                        text_predictions = [label_mapping.get(p, "Unknown") for p in predictions]
                        
                        # Add predictions to dataframe
                        result_df["Preprocessed_Text"] = preprocessed_texts
                        result_df["Label"] = text_predictions
                        result_df["Confidence"] = confidences
                        
                        # Calculate statistics
                        valid_predictions = [p for p in predictions if p is not None]
                        total_valid = len(valid_predictions)
                        if total_valid > 0:
                            positive_count = sum(1 for p in valid_predictions if p == 1)
                            negative_count = total_valid - positive_count
                            positive_percentage = (positive_count / total_valid) * 100
                            negative_percentage = (negative_count / total_valid) * 100
                        else:
                            positive_count = negative_count = 0
                            positive_percentage = negative_percentage = 0
                        
                        # Display success message
                        st.success(translate("File processed successfully!", current_language))
                        
                        # Show statistics
                        st.subheader(translate("Results Summary", current_language))
                        
                        # Display metrics in columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(translate("Total Texts", current_language), total_valid)
                        with col2:
                            st.metric(translate("Positive Texts", current_language), positive_count)
                        with col3:
                            st.metric(translate("Negative Texts", current_language), negative_count)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(translate("Positive Percentage", current_language), f"{positive_percentage:.2f}%")
                        with col2:
                            st.metric(translate("Negative Percentage", current_language), f"{negative_percentage:.2f}%")
                        
                        # Preview of processed data
                        st.subheader(translate("Preview of Processed Data", current_language))
                        st.dataframe(result_df.head(10))
                        
                        # Create download link
                        csv_filename = "sentiment_analysis_results.csv"
                        st.markdown(
                            get_csv_download_link(result_df, csv_filename, translate("Download Results", current_language)), 
                            unsafe_allow_html=True
                        )
                        
                except Exception as e:
                    st.error(f"{translate('Error processing file', current_language)}: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc(), language="python")
                        
        except Exception as e:
            st.error(f"{translate('Error processing file', current_language)}: {e}")