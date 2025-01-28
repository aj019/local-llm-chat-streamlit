from gtts import gTTS
import os
import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
import logging
import time
from PyPDF2 import PdfReader
import json
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize chat history and PDF content in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = ""

# Function to extract text from an uploaded PDF
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to stream chat response based on selected model
def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

# Function to read aloud text using gTTS and stream the audio directly
def read_aloud_gtts(text):
    try:
        tts = gTTS(text=text, lang="en")
        # Use BytesIO to handle the audio file in memory
        with BytesIO() as audio_io:
            tts.write_to_fp(audio_io)
            audio_io.seek(0)  # Move the seek pointer to the beginning of the file

            st.audio(audio_io.read(), format="audio/mp3", start_time=0)
            logging.info("Audio playback started.")
    except Exception as e:
        st.error("Error generating audio output.")
        logging.error(f"Error in read_aloud_gtts: {str(e)}")

# Function to export the chat history as a text or JSON file
def export_chat():
    chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
    chat_json = json.dumps(chat_history, indent=4)

    # Create a download link for the user
    st.download_button(
        label="Export Chat",
        data=chat_json,
        file_name="chat_history.json",
        mime="application/json"
    )

# Streamlit app setup
def main():
    st.title("PDFChat: Transforming PDFs into Interactive Conversations")
    logging.info("App started")

    # Sidebar for model selection and multiple PDF upload
    st.sidebar.header("Options")
    model = st.sidebar.selectbox("Choose a model", ["llama3.2", "phi3", "mistral", "gemma:7b", "deepseek-r1:7b,", "qwen2.5:7b"])
    logging.info(f"Model selected: {model}")

    uploaded_pdfs = st.sidebar.file_uploader("Attach PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_pdfs:
        st.session_state.pdf_content = ""
        for pdf_file in uploaded_pdfs:
            st.session_state.pdf_content += extract_pdf_text(pdf_file) + "\n\n"
        st.sidebar.success(f"{len(uploaded_pdfs)} PDF(s) uploaded and content extracted successfully!")

    # Button to export chat history (moved to sidebar)
    with st.sidebar:
        export_chat()

    # Display extracted PDF content (optional)
    if st.session_state.pdf_content:
        with st.expander("Show Extracted PDF Content"):
            st.text_area("PDF Content", st.session_state.pdf_content, height=200)

    # Prompt for user input and save to chat history
    if prompt := st.chat_input("Input question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        # Immediately display the user's query
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                # Start timing
                start_time = time.time()
                logging.info("Generating response")
                # Generate the bot's response with error handling for missing files
                with st.spinner("I am writing ...."):
                    try:
                        # Include PDF content as context for the LLM
                        pdf_context = st.session_state.pdf_content
                        full_prompt = f"Context from PDFs: {pdf_context}\n\nUser Query: {prompt}"

                        messages = [
                            ChatMessage(role=msg["role"], content=msg["content"])
                            for msg in st.session_state.messages
                        ]
                        messages.append(ChatMessage(role="user", content=full_prompt))

                        response_message = stream_chat(model, messages)

                        # Calculate the duration
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"

                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                        # Read the response aloud using gTTS
                        read_aloud_gtts(response_message)

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()