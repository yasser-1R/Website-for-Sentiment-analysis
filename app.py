import streamlit as st
import joblib
import os
import sys
import numpy as np
import pandas as pd
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
    "Arabic Tweets": {
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
    "Arabic Reviews": {
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
    "English Reviews": {
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
    "English Tweets": {
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
translations = {
    "Arabic": {
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
        "Text Analysis": "تحليل النص",
        "File Analysis": "تحليل الملف",
        "Upload CSV File": "تحميل ملف CSV",
        "Please upload a CSV file": "يرجى تحميل ملف CSV",
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
        "Preview of Processed Data": "معاينة البيانات المعالجة",
        "Select Column": "اختر العمود",
        "Home": "الرئيسية",
        "Welcome to Sentiment Analysis Tool": "مرحبًا بك في أداة تحليل المشاعر",
        "This tool helps you analyze sentiment in text data.": "تساعدك هذه الأداة على تحليل المشاعر في بيانات النص.",
        "Language": "اللغة",
        "English": "الإنجليزية",
        "Arabic": "العربية",
        "Switch to English": "تبديل إلى الإنجليزية",
        "Switch to Arabic": "تبديل إلى العربية",
        "Features:": "الميزات:",
        "Analyze sentiment in single texts": "تحليل المشاعر في النصوص الفردية",
        "Process entire CSV files": "معالجة ملفات CSV بالكامل",
        "Support for both Arabic and English text": "دعم للنص العربي والإنجليزي",
        "Multiple machine learning algorithms": "خوارزميات متعددة للتعلم الآلي",
        "How to Use:": "كيفية الاستخدام:",
        "Select Text Analysis or File Analysis from the sidebar": "اختر \"تحليل النص\" أو \"تحليل الملف\" من الشريط الجانبي",
        "Choose a dataset model (Arabic/English, Tweets/Reviews)": "اختر نموذج مجموعة البيانات (عربي/إنجليزي، تغريدات/مراجعات)",
        "Select a machine learning algorithm": "اختر خوارزمية التعلم الآلي",
        "Enter text or upload a CSV file for analysis": "أدخل نصًا أو قم بتحميل ملف CSV للتحليل"
    },
    "English": {
        # English translations are the keys themselves
    }
}

# Function to translate based on language
def translate(text, language):
    if language == "Arabic":
        return translations["Arabic"].get(text, text)
    return text

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# ========== Initialize session state ==========
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = list(model_mappings.keys())[0]
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = list(model_mappings[st.session_state.selected_dataset]["models"].keys())[0]
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'interface_language' not in st.session_state:
    st.session_state.interface_language = "English"

# ========== Sidebar Navigation ==========
st.sidebar.title("Sentiment Analysis")

# Language toggle button
current_language = st.session_state.interface_language
if current_language == "English":
    toggle_language = "Arabic"
    toggle_text = "Switch to Arabic"
else:
    toggle_language = "English"
    toggle_text = "Switch to English"

if st.sidebar.button(translate(toggle_text, current_language)):
    st.session_state.interface_language = toggle_language
    st.rerun()

st.sidebar.markdown("---")

# Navigation buttons
if st.sidebar.button(translate("Home", current_language)):
    st.session_state.page = "Home"
if st.sidebar.button(translate("Text Analysis", current_language)):
    st.session_state.page = "Text Analysis"
if st.sidebar.button(translate("File Analysis", current_language)):
    st.session_state.page = "File Analysis"

# Dataset and model selection (shared between Text and File Analysis)
if st.session_state.page in ["Text Analysis", "File Analysis"]:
    st.sidebar.markdown("---")
    
    # Dataset selection
    selected_dataset = st.sidebar.selectbox(
        translate("Select Dataset", current_language),
        list(model_mappings.keys()),
        index=list(model_mappings.keys()).index(st.session_state.selected_dataset)
    )
    
    # Update dataset in session state if changed
    if selected_dataset != st.session_state.selected_dataset:
        st.session_state.selected_dataset = selected_dataset
        # Reset model selection when dataset changes to prevent mismatching
        st.session_state.selected_model = list(model_mappings[selected_dataset]["models"].keys())[0]
    
    # Model selection
    selected_model_name = st.sidebar.selectbox(
        translate("Select Model Algorithm", current_language),
        list(model_mappings[selected_dataset]["models"].keys()),
        index=list(model_mappings[selected_dataset]["models"].keys()).index(st.session_state.selected_model)
    )
    
    # Update model in session state
    if selected_model_name != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_name
    
    # Load vectorizer and model
    try:
        vectorizer_path = os.path.join(model_mappings[selected_dataset]["folder"], model_mappings[selected_dataset]["vectorizer"])
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
        else:
            st.error(f"Vectorizer file not found: {vectorizer_path}")
            st.stop()
            
        model_path = os.path.join(model_mappings[selected_dataset]["folder"], model_mappings[selected_dataset]["models"][selected_model_name])
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.error(f"Model file not found: {model_path}")
            st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ========== Page Content ==========

# Home Page
if st.session_state.page == "Home":
    st.title(translate("Welcome to Sentiment Analysis Tool", current_language))
    st.markdown(translate("This tool helps you analyze sentiment in text data.", current_language))
    
    # Add more information about the tool
    st.markdown(f"### {translate('Features:', current_language)}")
    st.markdown(f"- {translate('Analyze sentiment in single texts', current_language)}")
    st.markdown(f"- {translate('Process entire CSV files', current_language)}")
    st.markdown(f"- {translate('Support for both Arabic and English text', current_language)}")
    st.markdown(f"- {translate('Multiple machine learning algorithms', current_language)}")
    
    st.markdown(f"### {translate('How to Use:', current_language)}")
    st.markdown(f"1. {translate('Select Text Analysis or File Analysis from the sidebar', current_language)}")
    st.markdown(f"2. {translate('Choose a dataset model (Arabic/English, Tweets/Reviews)', current_language)}")
    st.markdown(f"3. {translate('Select a machine learning algorithm', current_language)}")
    st.markdown(f"4. {translate('Enter text or upload a CSV file for analysis', current_language)}")

# Text Analysis Page
elif st.session_state.page == "Text Analysis":
    st.title(translate("Text Analysis", current_language))
    
    # Text Input
    user_input = st.text_area(
        translate("Enter text for sentiment analysis:", current_language),
        height=150,
        placeholder=translate("Type or paste text here...", current_language)
    )
    
    # Analyze Button
    if st.button(translate("Analyze Sentiment", current_language)):
        if user_input.strip() == "":
            st.warning(translate("Please enter some text.", current_language))
        else:
            try:
                # Get preprocessing function
                preprocess_fn = model_mappings[selected_dataset]["preprocessing"]
                
                with st.spinner(translate("Processing text...", current_language)):
                    # Preprocessing
                    preprocessed_text = preprocess_fn(user_input)
                    
                    # Display preprocessed text
                    st.subheader(translate("Preprocessed Text:", current_language))
                    st.text(preprocessed_text)
                    
                    if not preprocessed_text.strip():
                        st.warning(translate("Preprocessing resulted in empty text.", current_language))
                        st.stop()
                    
                    # Prediction
                    X_input = vectorizer.transform([preprocessed_text])
                    prediction = model.predict(X_input)[0]
                    
                    # Get prediction confidence
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
                        prediction_proba = None
                    
                    # Display prediction
                    label_mapping = {0: translate("Negative", current_language), 1: translate("Positive", current_language)}
                    prediction_label = label_mapping.get(prediction, "Unknown")
                    
                    st.subheader(translate("Predicted Sentiment:", current_language))
                    st.write(f"{'😊' if prediction == 1 else '😞'} {prediction_label}")
                    
                    # Show confidence
                    if confidence is not None:
                        st.write(f"{translate('Confidence', current_language)}: {confidence:.2f}%")
                        st.progress(confidence/100)
                    
                    # Show detailed probabilities
                    if prediction_proba is not None and len(prediction_proba) == 2:
                        st.subheader(translate("Sentiment Distribution", current_language))
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"{translate('Negative Probability', current_language)}: {prediction_proba[0]*100:.2f}%")
                        with col2:
                            st.write(f"{translate('Positive Probability', current_language)}: {prediction_proba[1]*100:.2f}%")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# File Analysis Page
elif st.session_state.page == "File Analysis":
    st.title(translate("File Analysis", current_language))
    
    # File uploader
    uploaded_file = st.file_uploader(
        translate("Please upload a CSV file", current_language),
        type=["csv"]
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # Try different encodings if UTF-8 fails
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            # Allow user to select which column to analyze
            if len(df.columns) > 0:
                selected_column = st.selectbox(
                    translate("Select Column", current_language),
                    df.columns
                )
                
                # Process button
                if st.button(translate("Analyze Sentiment", current_language)):
                    try:
                        with st.spinner(translate("Processing file...", current_language)):
                            # Get preprocessing function
                            preprocess_fn = model_mappings[selected_dataset]["preprocessing"]
                            
                            # Create a copy of the dataframe
                            result_df = df.copy()
                            
                            # Process each text and store results
                            preprocessed_texts = []
                            predictions = []
                            confidences = []
                            
                            # Create progress bar
                            progress_bar = st.progress(0)
                            total_rows = len(df)
                            
                            for idx, text in enumerate(df[selected_column]):
                                # Update progress bar
                                progress_bar.progress((idx + 1) / total_rows)
                                
                                if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
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
                            result_df["Sentiment"] = text_predictions
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
                            
                            # Preview processed data
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
            else:
                st.error("The uploaded CSV file is empty.")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")