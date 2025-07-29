import streamlit as st
import requests

st.set_page_config(page_title="🎬 YouTube Transcript QA Bot", page_icon="🎬")
st.title("🎬 YouTube Transcript QA Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "video_id" not in st.session_state:
    with st.expander("Step 1: Upload YouTube video"):
        youtube_url = st.text_input("Enter YouTube video URL:")
        if st.button("Ingest Transcript"):
            with st.spinner("Ingesting..."):
                r = requests.post("http://localhost:8000/transcript", json={"url": youtube_url})
                if r.status_code == 200:
                    st.success(r.json()['message'])
                    video_id = youtube_url.split("v=")[-1].split("&")[0]
                    st.session_state["video_id"] = video_id
                else:
                    st.error(r.json()["detail"])
else:
    st.success(f"Transcript for video ID `{st.session_state.video_id}` loaded successfully!")

    # Show chat interface
    st.subheader("💬 Chat with the Video")

    # Display past chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_input = st.chat_input("Ask a question about the video...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Show spinner while waiting for backend response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post("http://localhost:8000/ask", json={
                    "video_id": st.session_state["video_id"],
                    "query": user_input
                })

                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
                else:
                    error_message = response.json()["detail"]
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    st.error(f"Error: {error_message}")