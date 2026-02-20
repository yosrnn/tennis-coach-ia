import streamlit as st
import requests

st.title("Coach IA — Tennis Analytics")
st.caption("Système multi-agents RAG sur données ATP 2020-2024")

question = st.text_input("Pose ta question :", 
    placeholder="How does Djokovic perform on clay?")

if st.button("Analyser") and question:
    with st.spinner("Analyse en cours..."):
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question}
        )
        data = response.json()
        st.markdown(f"**Réponse :** {data['answer']}")
        st.caption(f"Agent utilisé : {data['agent_used']}")