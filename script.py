import nltk
# Download required nltk data if not already available
@st.cache_resource
def ensure_nltk_data():
    nltk.download('punkt')
    nltk.download('popular', quiet=True)
    nltk.download('nps_chat',quiet=True)
    nltk.download('punkt') 
    nltk.download('wordnet')
ensure_nltk_data()
