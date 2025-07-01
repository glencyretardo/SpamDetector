import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 🎀 Page settings
st.set_page_config(page_title="Spam Detector 💌", page_icon="🌸")

# 🎀 Header
st.markdown("<h1 style='text-align: center; color: hotpink;'>💌  Spam Detector 💌</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Try our spam detector… We promise to detect spam! 💫</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>By Julia Verzosa and Glency Retardo 💫</p>", unsafe_allow_html=True)

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

# 💬 User input with cute button
with st.form(key='spam_form'):
    user_input = st.text_input("💖 Enter a message to check for spam:", "")
    submit_button = st.form_submit_button("✨ Check Message ✨")

if submit_button and user_input:
    input_vec = vectorizer.transform([user_input.lower()])
    prediction = model.predict(input_vec)[0]
    st.markdown(f"<h3 style='color: {'red' if prediction == 'spam' else 'green'};'>🌼 Result: {prediction.upper()} 🌼</h3>", unsafe_allow_html=True)

# 📈 Optional: Show metrics
if st.checkbox("💅 Show model accuracy and report"):
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    st.write("🎯 **Accuracy:**", round(acc, 2))
    st.text("📄 Classification Report:")
    st.text(classification_report(y_test, y_pred))
