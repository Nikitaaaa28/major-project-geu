import streamlit as st
import requests  # We'll need this to talk to our server
import json

# --- Page Setup ---
st.set_page_config(
    page_title="SmartHealthChat",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 SmartHealthChat")
st.caption("Your friendly AI health information assistant")

# --- Chat History Setup ---
# We use st.session_state to store messages so they don't get lost
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm SmartHealthChat. How are you feeling today?"}
    ]

# --- Display all previous messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- The Chat Input Box ---
if prompt := st.chat_input("Ask me anything..."):
    # 1. Add the user's message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get the bot's response
    # This is the URL where your node server is running
    api_url = "http://localhost:3000/ask"
    
    try:
        # We send the user's prompt in the URL, e.g., .../ask?q=I have a fever
        response = requests.get(api_url, params={"q": prompt})

        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
        else:
            answer = f"Sorry, I couldn't connect to the server. Error: {response.status_code}"

    except requests.exceptions.RequestException as e:
        answer = "I'm sorry, I can't reach my brain right now. Please make sure the server is running."
        print(e) # For debugging

    # 3. Add the bot's response to history and display it
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
