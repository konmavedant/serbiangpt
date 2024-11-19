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
import pyperclip

# Ensure consistent language detection
DetectorFactory.seed = 0

# Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv())

# Streamlit page settings
st.set_page_config(page_title="Srpski.AI", page_icon="📸")
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

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
    """Displays the chat history in the sidebar with copy-to-clipboard functionality."""
    st.sidebar.subheader(
        "Chat History" if st.session_state.language == 'English' else "Istorija razgovora",
        help=(
            "Chats will only be stored for this session. If anything is important, please copy and save it elsewhere."
            if st.session_state.language == 'English'
            else "Razgovori će biti sačuvani samo za ovu sesiju. Ako je nešto važno, kopirajte i sačuvajte na drugom mestu."
        ),
    )

    # Iterate through chat history and display each message
    for i, message in enumerate(st.session_state.chat_history):
        st.sidebar.markdown(f"🧑 **You:** {message['human']}")

        # Display AI response in a text box with inline copy button
        response_key = f"response_{i}"  # Unique key for each text area
        ai_response = message['AI']
        st.sidebar.markdown(
            f"""
            <div style="position: relative; margin-bottom: 10px;">
                <textarea id="copy-target-{i}" style="width: 100%; height: 100px; padding-right: 40px;">{ai_response}</textarea>
                <button onclick="copyToClipboard('copy-target-{i}')" style="
                    position: absolute;
                    top: 0;
                    right: 0;
                    background-color: #f0f0f0;
                    border: none;
                    cursor: pointer;
                    padding: 5px 10px;
                ">📋</button>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Add JavaScript for the copy-to-clipboard functionality
    st.sidebar.markdown(
        """
        <script>
        function copyToClipboard(elementId) {
            const text = document.getElementById(elementId).value;
            navigator.clipboard.writeText(text).then(() => {
                alert("Copied to clipboard!");
            }).catch(err => {
                console.error("Failed to copy: ", err);
            });
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

def display_image_upload_options():
    """Display options for selecting between camera and file uploads."""
    st.markdown("**📷 Choose an Option for Image Capture**" if st.session_state.language == 'English' else "📷 Izaberite opciju za snimanje slike", help = "You can either click a photo using your camera or upload an existing image from your device." if st.session_state.language == 'English' else "Možete ili snimiti fotografiju kamerom ili preneti postojeću sliku sa svog uređaja.")

    # Create two buttons side-by-side using columns
    col1, col2 = st.columns([1, 1])

    # Initialize session state variables to track selection
    if 'show_camera' not in st.session_state:
        st.session_state.show_camera = False
    if 'show_uploader' not in st.session_state:
        st.session_state.show_uploader = False

    # Camera button
    with col1:
        if st.button("📸 Open Camera" if st.session_state.language == 'English' else "📸 Otvori Kameru"):
            st.session_state.show_camera = True
            st.session_state.show_uploader = False

    # File uploader button
    with col2:
        if st.button("📁 Upload File" if st.session_state.language == 'English' else "📁 Otpremi Fajl"):
            st.session_state.show_camera = False
            st.session_state.show_uploader = True

    # Display camera input or file uploader based on selection
    uploaded_file_camera = None
    uploaded_file = None

    if st.session_state.show_camera:
        st.markdown("**📸 Use Camera to Capture Image**" if st.session_state.language == 'English' else "**📸 Koristite kameru za snimanje slike**")
        uploaded_file_camera = st.camera_input("Click to take a photo" if st.session_state.language == 'English' else "Kliknite da uslikate")

    if st.session_state.show_uploader:
        st.markdown("**📁 Upload Image File**" if st.session_state.language == 'English' else "**📁 Otpremi sliku**")
        uploaded_file = st.file_uploader("Choose an image file" if st.session_state.language == 'English' else "Izaberite fajl sa slikom", type=["jpg", "jpeg", "png"])

    return uploaded_file_camera, uploaded_file

st.markdown(
    """
    <style>
        /* Ensure buttons remain side-by-side on small screens */
        @media (max-width: 600px) {
            .stButton>button {
                width: 100%;
            }
        }

        /* Adjust column layout on smaller screens for better appearance */
        .stButton {
            display: inline-block;
            width: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)




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

    # Handle image processing
    if uploaded_file_camera is not None or uploaded_file is not None:
        uploaded_file = uploaded_file_camera or uploaded_file

        with st.spinner("Processing image for text..." if st.session_state.language == 'English' else "Obrada slike za tekst..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                # Perform OCR
                ocr_text, translated_text, detected_language, target_language = perform_ocr_with_vision_api(temp_path, credentials)
                
                # Append OCR results to chat history
                user_action = "Uploaded Image for OCR Translation"
                st.session_state.chat_history.append({
                    "human": user_action,
                    "AI": (
                        f"**Detected Text ({detected_language.capitalize()}):** {ocr_text}\n\n"
                        f"**Translated to {target_language.capitalize()}:** {translated_text}"
                    )
                })

                # Display chat response for OCR result
                with st.chat_message("user"):
                    st.markdown(user_action)
                with st.chat_message("assistant"):
                    st.markdown(
                        f"**Detected Text ({detected_language.capitalize()}):** {ocr_text}\n\n"
                        f"**Translated to {target_language.capitalize()}:** {translated_text}"
                    )

                # Save translated text to session state for further use
                st.session_state.ocr_text = translated_text

            finally:
                os.remove(temp_path)

    # Handle user question input
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
