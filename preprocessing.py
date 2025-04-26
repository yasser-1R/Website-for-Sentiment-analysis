import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ========== General Utilities ==========
def tokenization(text):
    if isinstance(text, str):
        return re.findall(r'\S+', text)
    return []

def rejoin_tokens(tokens):
    filtered_tokens = [token for token in tokens if isinstance(token, str) and token.strip()]
    return ' '.join(filtered_tokens)

# ========== Arabic Preprocessing Utilities ==========
def remove_non_arabic(text):
    if isinstance(text, str):
        return re.sub(r'[^\u0620-\u064A\s]', ' ', text)
    return ''

def remove_arabic_stopwords(tokens):
    arabic_stopwords = set(stopwords.words("arabic"))
    return [token for token in tokens if token not in arabic_stopwords]

def get_root_ar(word, stemmer):
    if isinstance(word, str):
        return stemmer.stem(word)
    return word

def stemming_AR(tokens):
    stemmer = ISRIStemmer()
    return [get_root_ar(token, stemmer) for token in tokens]

# ========== English Preprocessing Utilities ==========
def remove_user_tags(text):
    if isinstance(text, str):
        text = re.sub(r'@ \S+', '', text)
        text = re.sub(r'@\S+', '', text)
        return text
    return ''

def remove_pic_noise(text):
    if isinstance(text, str):
        return re.sub(r'pic\.\S+(\s*/\s*\S+)?', '', text).strip()
    return text

def remove_br_tags(text):
    if isinstance(text, str):
        return text.replace('<br />', ' ')
    return ''

def convert_to_lowercase(text):
    if isinstance(text, str):
        return text.lower()
    return ''

def remove_non_latin(text):
    if isinstance(text, str):
        return re.sub(r'[^a-z\s]', ' ', text)
    return ''

def remove_ENG_stopwords(tokens):
    if isinstance(tokens, list):
        return [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return []

def get_root_en(word, stemmer):
    if isinstance(word, str):
        return stemmer.stem(word)
    return word

def stemming_ENG(tokens):
    stemmer = PorterStemmer()
    return [get_root_en(token, stemmer) for token in tokens]

# ========== Dataset-specific Preprocessing Functions ==========
def arabic_preprocessing(text):
    """Main Arabic preprocessing function that combines all Arabic preprocessing steps"""
    if isinstance(text, str):
        text = remove_non_arabic(text)
        tokens = tokenization(text)
        tokens = remove_arabic_stopwords(tokens)
        tokens = stemming_AR(tokens)
        return rejoin_tokens(tokens)
    return ''

def english_preprocessing(text):
    """Main English preprocessing function that combines all English preprocessing steps"""
    if isinstance(text, str):
        text = remove_br_tags(text)
        text = remove_user_tags(text)
        text = remove_pic_noise(text)
        text = convert_to_lowercase(text)
        text = remove_non_latin(text)
        tokens = tokenization(text)
        tokens = remove_ENG_stopwords(tokens)
        tokens = stemming_ENG(tokens)
        return rejoin_tokens(tokens)
    return ''

# Dataset-specific preprocessors (maintained for backwards compatibility)
def ASTD_pretraitement(text):
    return arabic_preprocessing(text)

def LABR_pretraitement(text):
    return arabic_preprocessing(text)

def IMDB_pretraitement(text):
    return english_preprocessing(text)

def TESA_pretraitement(text):
    return english_preprocessing(text)