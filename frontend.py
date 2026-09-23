"""
Run:
    streamlit run frontend.py
"""

import os
import streamlit as st
from pinecone import Pinecone
from backend import ask_question, get_index

st.set_page_config(page_title="Resume Chat", layout="centered")
st.title("Resume Chat")
st.page_link("https://drive.google.com/file/d/1I_I3v9vI3Z1OTwfQuwmksIxUGIpf5sa3/view?usp=sharing", label="View resume", icon="📄")
st.caption("Ask questions about my resume")

index = get_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about my resume..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching resume..."):
            answer = ask_question(
                question=prompt,
                index=index,
            )
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})