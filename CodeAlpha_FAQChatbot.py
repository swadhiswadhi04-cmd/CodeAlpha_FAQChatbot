import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 FAQ Chatbot")

faq_questions = [
    "What is your return policy?",
    "How can I contact support?",
    "What payment methods are accepted?",
    "How long does shipping take?",
    "Do you offer refunds?"
]

faq_answers = [
    "You can return products within 30 days of purchase.",
    "You can contact support via email at support@example.com.",
    "We accept credit cards, debit cards, and UPI payments.",
    "Shipping usually takes 5-7 business days.",
    "Yes, refunds are available within 30 days."
]

user_question = st.text_input("Ask your question:")

if user_question:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(faq_questions + [user_question])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_index = similarity.argmax()

    st.write("### Answer:")
    st.success(faq_answers[best_match_index])
