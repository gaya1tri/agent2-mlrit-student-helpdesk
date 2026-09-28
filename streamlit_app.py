import streamlit as st
from app import helpdesk_graph

st.title("MLRIT Student Helpdesk Assistant")

question = st.text_input("Ask your question")

if st.button("Ask") and question:
    result = helpdesk_graph.invoke({
        "question": question
    })

    st.subheader("Answer")
    st.write(result["answer"])