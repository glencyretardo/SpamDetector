import pandas as pd
import streamlit as st
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 🎀 Page settings
st.set_page_config(page_title="Spam Detector 💌", page_icon="🌸")

# 🌸 Custom background and styling
st.markdown("""
    <style>
    .main {
        background-color: #fff0f5;
        font-family: "Comic Sans MS", cursive, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# 🎀 Header
st.markdown("<h1 style='text-align: center; color: hotpink;'>💌  Spam Detector 💌</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Try our spam detector… We promise to detect spam! 💫</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>By Glency Retardo and Julia Verzosa 💫</p>", unsafe_allow_html=True)

# 📘 Sidebar info
with st.sidebar:
    st.title("📘 About")
    st.write("This is a machine learning-based spam message detector built using Streamlit and scikit-learn.")
    st.write("Developed by **Glency Retardo** and **Julia Verzosa**.")
    st.markdown("---")
    st.write("🛠️ Model: Multinomial Naive Bayes")
    st.write("📚 Dataset: SMS Spam Collection")
    st.write("💡 ML concepts: NLP, CountVectorizer, Supervised Learning")

# 📊 Load and clean data
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].str.lower()

# ✨ Split & vectorize
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🧠 Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 🧠 Initialize session state for message history
if "history" not in st.session_state:
    st.session_state.history = []

# 💬 User input with cute form button
with st.form(key='spam_form'):
    user_input = st.text_input("💖 Enter a message to check for spam:")
    submit_button = st.form_submit_button("✨ Check Message ✨")

if submit_button and user_input:
    with st.spinner("Analyzing message..."):
        time.sleep(1.5)
        input_vec = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_vec)[0]
        probs = model.predict_proba(input_vec)[0]
        confidence = max(probs)

        emoji = "🚫📩" if prediction == "spam" else "✅💌"
        color = "red" if prediction == "spam" else "green"
        st.markdown(f"<h3 style='color:{color};'>{emoji} Result: {prediction.upper()} {emoji}</h3>", unsafe_allow_html=True)
        st.write(f"🔎 Model confidence: **{confidence:.2f}**")

        # Save to history
        st.session_state.history.append((user_input, prediction))

# 🕒 Show message history
if st.checkbox("🕒 Show message history"):
    st.subheader("📜 Past Messages Checked")
    for msg, pred in reversed(st.session_state.history[-5:]):  # Show last 5
        color = "red" if pred == "spam" else "green"
        st.markdown(f"<p style='color:{color};'>💬 {msg} → <b>{pred.upper()}</b></p>", unsafe_allow_html=True)

# 📈 Optional: Show metrics
if st.checkbox("📊 Show model accuracy and report"):
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    st.write("🎯 **Accuracy:**", round(acc, 2))
    st.text("📄 Classification Report:")
    st.text(classification_report(y_test, y_pred))
