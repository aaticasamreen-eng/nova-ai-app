import os
import tempfile
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# Page configuration
st.set_page_config(page_title="Nova AI - Chat & Export", page_icon="🤖", layout="centered")
st.title("🤖 Nova AI Assistant")
st.caption("Powered by Google Gemini (Multimodal & Streaming)")

# Validate API Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY environment variable is not set. Please set it and restart.")
    st.stop()

# Initialize Client
if "client" not in st.session_state:
    st.session_state.client = genai.Client()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to compile message history into a downloadable string
def get_chat_text():
    formatted_chat = "--- Nova AI Chat History ---\n\n"
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Nova"
        formatted_chat += f"[{role}]:\n{msg['content']}\n\n"
    return formatted_chat

# Sidebar Controls
with st.sidebar:
    st.header("Options & Files")
    uploaded_file = st.file_uploader(
        "Attach an Image or PDF", 
        type=["pdf", "jpg", "jpeg", "png", "webp"]
    )
    
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            st.success(f"📄 Attached PDF: **{uploaded_file.name}**")
        else:
            img_preview = Image.open(uploaded_file)
            st.image(img_preview, caption="Attached Image", use_container_width=True)
        
    st.divider()

    # Download Button for Exporting Chat History
    if st.session_state.messages:
        st.download_button(
            label="📥 Download Chat (.txt)",
            data=get_chat_text(),
            file_name="nova_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Ask Nova something..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            contents = []
            temp_file_path = None

            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_file_path = tmp.name

                    gemini_file = st.session_state.client.files.upload(file=temp_file_path)
                    contents.append(gemini_file)
                else:
                    image_data = Image.open(uploaded_file)
                    contents.append(image_data)

            contents.append(prompt)

            # Stream response generator
            def stream_response():
                stream = st.session_state.client.models.generate_content_stream(
                    model="gemini-3.5-flash-lite",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful, witty, and concise AI assistant named Nova."
                    )
                )
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text

            full_response = st.write_stream(stream_response())
            
            # Save completed text to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")
        
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)