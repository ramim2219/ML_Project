import streamlit as st
import torch
import torch.nn as nn
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.nn.utils.rnn import pad_sequence

# ===================== NLTK Downloads =====================
# Ensure NLTK data is downloaded (only needed if not present)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ===================== CONSTANTS (Must match training) =====================
MAX_LEN = 150  # Max sequence length used during training
VOCAB_SIZE = 20000 # Vocab size used during training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== GRU Model Definition (Must match training) =====================
class GRUSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2, bidirectional=True, dropout=0.5, num_classes=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        multiplier = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_dim * multiplier, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)

        if self.gru.bidirectional:
            # Concatenate the last hidden states from forward and backward directions
            # This was slightly different in the notebook, using max pooling over time.
            # For consistency, I'll use max pooling as specified in the original GRU model's forward pass comments.
            # last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            pass # Max pooling is used below
        else:
            # last_hidden = hidden[-1,:,:]
            pass # Max pooling is used below

        # Use max pooling over time (as in notebook)
        pooled = torch.max(gru_out, dim=1)[0]

        x = self.dropout(pooled)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# ===================== LOAD MODEL & TOKENIZER =====================
import torch.serialization
import torch.nn as nn

@st.cache_resource
def load_model_and_vocab():
    # Load word2idx
    with open("word2idx.pkl", "rb") as handle:
        word2idx = pickle.load(handle)

    # Initialize the model with the exact same parameters as trained
    model = GRUSentimentClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=300,
        hidden_dim=256,
        num_layers=2,
        bidirectional=True,
        dropout=0.5,
        num_classes=3
    ).to(device)

    # Add the GRUSentimentClassifier and other necessary layers to safe globals
    torch.serialization.add_safe_globals([GRUSentimentClassifier, nn.Embedding, nn.GRU])

    # Load the trained model state_dict (or full model)
    model.load_state_dict(torch.load("final_gru_sentiment_model.pth", map_location=device))
    model.eval()  # Set to evaluation mode
    return word2idx, model


word2idx, model = load_model_and_vocab()

# ===================== CLEANING & Preprocessing (Must match training) =====================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()  # Ensure text is string and lowercase
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with single, strip whitespace
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 1]
    return ' '.join(words)

def text_to_indices(text):
    return [word2idx.get(word, word2idx.get('<unk>', 1)) for word in text.split()[:MAX_LEN]]

def preprocess_text(text):
    cleaned = clean_text(text)
    indexed = text_to_indices(cleaned)

    # Pad the sequence
    if len(indexed) < MAX_LEN:
        padded_review = indexed + [word2idx.get('<pad>', 0)] * (MAX_LEN - len(indexed))
    else:
        padded_review = indexed[:MAX_LEN] # Truncate if longer than MAX_LEN

    return torch.tensor([padded_review], dtype=torch.long).to(device)

# ===================== UI LAYOUT =====================
st.set_page_config(layout="centered", page_title="Sentiment Analyzer")
st.markdown("""
<style>
    .main-container { padding: 20px; max-width: 800px; margin: auto; font-family: 'Arial', sans-serif; }
    .app-title { font-size: 2.5em; color: #4CAF50; text-align: center; margin-bottom: 10px; }
    .app-subtitle { font-size: 1.2em; color: #555; text-align: center; margin-bottom: 30px; }
    .card { background-color: #f9f9f9; border-radius: 8px; padding: 25px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .stButton > button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 20px; font-size: 1.1em; border: none; cursor: pointer; transition: background-color 0.3s; }
    .stButton > button:hover { background-color: #45a049; }
    .positive { color: green; font-weight: bold; }
    .negative { color: red; font-weight: bold; }
    .neutral { color: orange; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='app-title'>Sentiment Analyzer ⚡</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Analyze customer reviews using an advanced GRU neural network</div>", unsafe_allow_html=True)

# ===================== INPUT CARD =====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
user_input = st.text_area(
    "Enter your review:",
    height=150,
    placeholder="Example: The product quality was amazing and delivery was super fast! ❤️"
)
analyze_btn = st.button("Analyze Sentiment 🔍")
st.markdown("</div>", unsafe_allow_html=True)

# ===================== RESULT =====================
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter a review first!")
    else:
        with st.spinner('Analyzing sentiment...'):
            X_input_tensor = preprocess_text(user_input)

            with torch.no_grad():
                outputs = model(X_input_tensor)
                # Get probabilities (optional, softmax can be applied here if needed for scores)
                # probabilities = torch.softmax(outputs, dim=1)
                # predicted_prob = probabilities[0].max().item()

                predicted_class = outputs.argmax(dim=1).item()

            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            label = sentiment_map[predicted_class]

            # For styling the output
            sentiment_class_css = label.lower()

            st.markdown(f"""
            <div class='card'>
                <h3>Prediction: <span class='{sentiment_class_css}'>{label}</span></h3>
                <!-- Optional: Display confidence if needed -->
                <!-- <p>Confidence: {predicted_prob:.2f}</p> -->
            </div>
            """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
