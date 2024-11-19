import os
import json
import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import requests
import streamlit.components.v1 as components


# Ensure consistent language detection
DetectorFactory.seed = 0

# Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv())

# Streamlit page settings
st.set_page_config(page_title="Srpski.AI", page_icon="📸")

def load_credentials_from_url(json_url):
    """Load service account credentials from a URL."""
    try:
        response = requests.get(json_url)
        response.raise_for_status()
        credentials_json = response.json()
        credentials = service_account.Credentials.from_service_account_info(credentials_json)
        return credentials
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to load credentials from URL: {e}")

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model1' not in st.session_state:
        st.session_state.model1 = 'llama3-70b-8192'
    if 'model2' not in st.session_state:
        st.session_state.model2 = 'gemma-7b-it'
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'ocr_text' not in st.session_state:
        st.session_state.ocr_text = ""

def initialize_groq_chat(groq_api_key, model_name):
    """Initialize Groq chat with API key and model."""
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

def perform_ocr_with_vision_api(image_path, credentials):
    """Performs OCR using Google Cloud Vision API with the provided credentials."""
    client = vision.ImageAnnotatorClient(credentials=credentials)
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    ocr_text = response.text_annotations[0].description if response.text_annotations else ""
    detected_language = 'unknown'
    translated_text = ocr_text
    target_language = 'Unknown'

    # Attempt to detect language from Google Vision's locale
    locale = response.text_annotations[0].locale if response.text_annotations else None
    if locale:
        detected_language = locale
    else:
        try:
            detected_language = detect(ocr_text)
        except Exception as e:
            print(f"Language detection failed: {e}")
            detected_language = "unknown"

    try:
        # Handle translation
        if detected_language in ['sr', 'mk', 'unknown']:
            translated_text = GoogleTranslator(source="auto", target="en").translate(ocr_text)
            target_language = "English"
        elif detected_language == 'en':
            translated_text = GoogleTranslator(source="en", target="sr").translate(ocr_text)
            target_language = "Serbian"
        else:
            translated_text = f"Text detected as {detected_language}, unsupported for translation."
            target_language = "Unknown"
    except Exception as e:
        translated_text = f"Translation failed. Extracted text: {ocr_text}"
        print(f"Error during translation: {e}")

    return ocr_text, translated_text, detected_language, target_language

def initialize_conversation(groq_chat, memory):
    """Initialize conversation chain with memory."""
    return ConversationChain(llm=groq_chat, memory=memory)

def process_user_question(user_question, conversation1, conversation2, uploaded_image=None, ocr_text=""):
    """Processes the user question and generates a hybrid response."""
    user_question_for_model = user_question
    if st.session_state.language == 'Serbian':
        user_question_for_model = "Molim vas, odgovarajte na srpskom: " + user_question

    if uploaded_image and ocr_text:
        user_question_for_model += f" (Tekst iz slike: {ocr_text})"

    response1 = conversation1(user_question_for_model).get('response', '').strip()
    response2 = conversation2(user_question_for_model).get('response', '').strip()

    hybrid_response = response1 if response1 == response2 else f"{response1} {response2}"

    if not st.session_state.chat_history or st.session_state.chat_history[-1]['human'] != user_question:
        st.session_state.chat_history.append({'human': user_question, 'AI': hybrid_response})
    else:
        st.session_state.chat_history[-1]['AI'] = hybrid_response

def display_chat_history():
    """Displays the chat history in the sidebar.""" 
    st.sidebar.subheader("Chat History" if st.session_state.language == 'English' else "Istorija razgovora")
    for message in st.session_state.chat_history:
        st.sidebar.markdown(f"🧑 You: {message['human']}")
        if message['AI']:
            st.sidebar.markdown(f"🤖 AI: {message['AI']}\n")

def display_image_upload_options():
    """Display options for camera and file uploads."""
    st.markdown("**📷 Click to Open Camera for Image Capture**")
    if st.button("Open Camera"):
        # Embedding custom JS to open the camera in full-screen with front/back switch functionality
        components.html("""
            <style>
                #camera-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: black;
                }
                #camera-container video {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                #controls {
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    color: white;
                }
                #photo-button {
                    position: absolute;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    padding: 10px;
                    background-color: rgba(0, 0, 0, 0.7);
                    color: white;
                    border-radius: 5px;
                    cursor: pointer;
                }
            </style>
            <div id="camera-container">
                <video id="video" autoplay></video>
                <div id="controls">
                    <button id="switch-camera" style="background-color: rgba(0, 0, 0, 0.7); color: white; padding: 10px; cursor: pointer;">Switch Camera</button>
                </div>
                <button id="photo-button" onclick="takePhoto()">Take Photo</button>
            </div>
            <script>
                let videoElement = document.getElementById('video');
                let switchCameraButton = document.getElementById('switch-camera');
                let currentStream = null;
                let isFrontCamera = false;

                // Initialize the camera
                function startCamera() {
                    navigator.mediaDevices.enumerateDevices().then(devices => {
                        const videoDevices = devices.filter(device => device.kind === 'videoinput');
                        if (videoDevices.length > 0) {
                            const constraints = { video: { facingMode: isFrontCamera ? 'user' : 'environment' } };
                            navigator.mediaDevices.getUserMedia(constraints)
                                .then((stream) => {
                                    if (currentStream) {
                                        currentStream.getTracks().forEach(track => track.stop());
                                    }
                                    currentStream = stream;
                                    videoElement.srcObject = stream;
                                }).catch((err) => {
                                    alert("Unable to access camera: " + err.message);
                                });
                        }
                    });
                }

                function takePhoto() {
                    const canvas = document.createElement('canvas');
                    canvas.width = videoElement.videoWidth;
                    canvas.height = videoElement.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                    const dataUrl = canvas.toDataURL('image/png');
                    alert("Photo captured! Data URL: " + dataUrl);
                }

                switchCameraButton.addEventListener('click', () => {
                    isFrontCamera = !isFrontCamera;
                    startCamera();
                });

                startCamera();
            </script>
        """, height=600)

    uploaded_file_camera = None  # Initially, no image captured

    st.markdown("**🖼️ Or Upload Image File**")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    return uploaded_file_camera, uploaded_file


def main():
    # URL to the public JSON credentials file hosted on Google Cloud Storage
    credentials_url = "https://storage.googleapis.com/serbia-gpt/gentle-impulse-442016-m5-685a326dc711.json"
    
    try:
        credentials = load_credentials_from_url(credentials_url)
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return

    # Initialize session state
    initialize_session_state()

    # Streamlit UI for language toggle
    language_toggle = st.toggle("Switch to Serbian")
    st.session_state.language = 'Serbian' if language_toggle else 'English'

    # Set page title and instructions
    title_text = "Srpski.AI 📸"
    st.title(title_text)
    welcome_text = (
        "Translate any photo from Serbian to English or vice versa and chat with Srpski.AI!"
        if st.session_state.language == 'English'
        else "Prevedite bilo koju fotografiju sa srpskog na engleski ili obrnuto i razgovarajte sa Srpski.AI!"
    )
    st.markdown(welcome_text)

    memory = ConversationBufferWindowMemory(k=10)
    display_chat_history()
    st.divider()

    # Display options for image upload
    uploaded_file_camera, uploaded_file = display_image_upload_options()

    if uploaded_file_camera is not None:
        with st.spinner("Processing image for text..." if st.session_state.language == 'English' else "Obrada slike za tekst..."):
            temp_path = f"temp_{uploaded_file_camera.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file_camera.getbuffer())
            try:
                ocr_text, translated_text, detected_language, target_language = perform_ocr_with_vision_api(temp_path, credentials)
                
                st.session_state.ocr_text = translated_text
                st.success(
                    f"Text detected in {detected_language.capitalize()} and translated to {target_language.capitalize()}!"
                    if st.session_state.language == 'English'
                    else f"Tekst prepoznat na {detected_language.capitalize()} i preveden na {target_language.capitalize()}!"
                )
                st.markdown(f"*Extracted Text ({detected_language.capitalize()}):* {ocr_text}\n\n*Translated to {target_language.capitalize()}:* {translated_text}")
            finally:
                os.remove(temp_path)

    elif uploaded_file is not None:
        with st.spinner("Processing image for text..." if st.session_state.language == 'English' else "Obrada slike za tekst..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                ocr_text, translated_text, detected_language, target_language = perform_ocr_with_vision_api(temp_path, credentials)
                
                st.session_state.ocr_text = translated_text
                st.success(
                    f"Text detected in {detected_language.capitalize()} and translated to {target_language.capitalize()}!"
                    if st.session_state.language == 'English'
                    else f"Tekst prepoznat na {detected_language.capitalize()} i preveden na {target_language.capitalize()}!"
                )
                st.markdown(f"*Extracted Text ({detected_language.capitalize()}):* {ocr_text}\n\n*Translated to {target_language.capitalize()}:* {translated_text}")
            finally:
                os.remove(temp_path)

    if user_question := st.chat_input("Ask Questions" if st.session_state.language == 'English' else "Postavite pitanja"):
        if not st.session_state.chat_history or st.session_state.chat_history[-1]["human"] != user_question:
            st.session_state.chat_history.append({"human": user_question, "AI": ""})
        with st.chat_message("user"):
            st.markdown(user_question)

        groq_chat_model1 = initialize_groq_chat(os.environ['GROQ_API_KEY'], st.session_state.model1)
        groq_chat_model2 = initialize_groq_chat(os.environ['GROQ_API_KEY'], st.session_state.model2)
        conversation_model1 = initialize_conversation(groq_chat_model1, memory)
        conversation_model2 = initialize_conversation(groq_chat_model2, memory)

        with st.spinner("Bot is thinking..." if st.session_state.language == 'English' else "Četbot razmišlja..."):
            process_user_question(
                user_question,
                conversation_model1,
                conversation_model2,
                uploaded_image=st.session_state.uploaded_image,
                ocr_text=st.session_state.ocr_text
            )

        ai_response = st.session_state.chat_history[-1]["AI"]
        with st.chat_message("assistant"):
            st.markdown(ai_response)

if __name__ == "__main__":
    main()
