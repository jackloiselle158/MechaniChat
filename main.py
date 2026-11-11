import asyncio
import streamlit as st

from backend.chat_service import get_complete_response
from backend.retrieval import collection


# Temporary Streamlit history while the React interface is being connected
def get_session_history_fn(session_id):
    if "memory_storage" not in st.session_state:
        st.session_state.memory_storage = {}
    if session_id not in st.session_state.memory_storage:
        st.session_state.memory_storage[session_id] = []
    return st.session_state.memory_storage[session_id]


def convert_streamlit_history(session_history):
    return list(session_history)


# Function to use the same backend service from the temporary Streamlit UI
def process_with_calls(user_input):
    session_history = get_session_history_fn("chat")
    history = convert_streamlit_history(session_history)
    response, citations = asyncio.run(
        get_complete_response(user_input, history)
    )
    session_history.append({"role": "user", "content": user_input})
    session_history.append({"role": "assistant", "content": response})
    return response, citations


# Streamlit interface kept temporarily while the React app is tested
print(collection.count())
st.set_page_config(layout="centered")
st.title("MechaniChat")
user_prompt = st.text_area("Ask a question:")

if st.button("Ask MechaniChat"):
    response, citations = process_with_calls(user_prompt)
    st.subheader("Response:")
    st.write(response)
    if citations:
        st.subheader("Sources:")
        for citation in citations:
            st.write(f"[{citation.id}] {citation.title}")
