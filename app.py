import streamlit as st
import os
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
import toml

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

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "credentials_verified" not in st.session_state:
    st.session_state.credentials_verified = False

# Default credentials
DEFAULT_USERNAME = "user_name"
DEFAULT_PASSWORD = "user@123"

def login():
    """Display the login content with Verify and Login buttons."""
    st.title("Login to Srpski.AI")
    
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    # Verify button
    if st.button("Verify"):
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            st.session_state.credentials_verified = True
            st.success("Valid credentials!")
        else:
            st.session_state.credentials_verified = False
            st.error("Invalid username or password")

    # Login button (enabled only if credentials are verified)
    if st.session_state.credentials_verified:
        if st.button("Login"):
            st.session_state.logged_in = True


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

def setup_sidebar_language_selection():
    """Setup the sidebar for selecting translation language."""
    st.sidebar.title("Language Options", help = "Select the language for Image Translation")
    #st.sidebar.info("Select the language for translation.")

    # Create an instance of GoogleTranslator to access supported languages
    translator = GoogleTranslator()
    available_languages = translator.get_supported_languages(as_dict=False)
    
    # Add a dropdown for language selection
    selected_language = st.sidebar.selectbox(
        "Translate to (Select a Language):", 
        options=available_languages, 
        index=available_languages.index("english")  # Default to English
    )

    # Save selected language in session state
    st.session_state.selected_translation_language = selected_language
    return selected_language

def perform_ocr_with_vision_api(image_path, credentials, selected_language):
    """Performs OCR and translates based on selected language."""
    client = vision.ImageAnnotatorClient(credentials=credentials)
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    ocr_text = response.text_annotations[0].description if response.text_annotations else ""
    detected_language = 'unknown'

    # Attempt to detect the language of the OCR text
    locale = response.text_annotations[0].locale if response.text_annotations else None
    if locale:
        detected_language = locale
    else:
        try:
            detected_language = detect(ocr_text)
        except Exception as e:
            print(f"Language detection failed: {e}")
            detected_language = "unknown"

    # Translation logic
    translated_text = ocr_text  # Default to no translation
    try:
        # Perform translation if the detected language differs from selected language
        if detected_language != selected_language:
            translated_text = GoogleTranslator(source="auto", target=selected_language).translate(ocr_text)
        else:
            translated_text = f"Text is already in {selected_language}."
    except Exception as e:
        translated_text = f"Translation failed. Extracted text: {ocr_text}"
        print(f"Translation error: {e}")

    return ocr_text, translated_text, detected_language, selected_language

def initialize_conversation(groq_chat, memory):
    """Initialize conversation chain with memory."""
    return ConversationChain(llm=groq_chat, memory=memory)

def process_user_question(user_question, conversation1, conversation2, uploaded_image=None, ocr_text=""): 
    """Processes the user question and generates a hybrid response."""
    # Always prefix user questions with a directive to answer in Serbian if the toggle is on
    if st.session_state.language == 'Serbian':
        user_question_for_model = f"Molim vas, odgovarajte na srpskom: {user_question}"
    else:
        user_question_for_model = user_question

    # Append OCR text if available
    if uploaded_image and ocr_text:
        user_question_for_model += f" (Tekst iz slike: {ocr_text})"

    # Get responses from both conversation models
    response1 = conversation1(user_question_for_model).get('response', '').strip()
    response2 = conversation2(user_question_for_model).get('response', '').strip()

    # Generate hybrid response
    hybrid_response = response1 if response1 == response2 else f"{response1} {response2}"

    # Append the conversation to chat history
    if not st.session_state.chat_history or st.session_state.chat_history[-1]['human'] != user_question:
        st.session_state.chat_history.append({'human': user_question, 'AI': hybrid_response})
    else:
        st.session_state.chat_history[-1]['AI'] = hybrid_response

def display_chat_history():
    """Displays the chat history in the sidebar with a copy-to-clipboard option for AI responses."""
    
    # Custom CSS to style the sidebar and code block
    st.markdown(
        """
        <style>
        /* Custom CSS for the sidebar and code block */
        .sidebar .code {
            width: 100% !important; /* Full width inside the sidebar */
            height: auto !important; /* Adjust height based on content */
            white-space: pre-wrap; /* Ensures text wrapping */
            word-wrap: break-word; /* Break words to fit the box */
            background-color: #f7f7f7; /* Light background */
            padding: 10px; /* Space around text */
            border-radius: 8px; /* Rounded corners */
            font-family: 'Courier New', monospace; /* Monospace font */
            font-size: 14px; /* Adjust the font size */
            overflow-x: auto; /* Horizontal scroll if the text is too wide */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Display Chat History
    st.sidebar.subheader(
        "Chat History" if st.session_state.language == 'English' else "Istorija razgovora",
        help=(
            "Chats will only be stored for this session. If anything is important, please copy and save it elsewhere."
            if st.session_state.language == 'English'
            else "Razgovori će biti sačuvani samo za ovu sesiju. Ako je nešto važno, kopirajte i sačuvajte na drugom mestu."
        ),
    )

    for i, message in enumerate(st.session_state.chat_history):
        # Display user input
        st.sidebar.markdown(f"🧑 **You:** {message['human']}")

        # Display AI response with copy-to-clipboard option
        ai_response = message['AI']
        st.sidebar.markdown("🤖 **AI Response:**")
        
        # Code block with custom CSS styles applied
        st.sidebar.code(ai_response, language="text")

def display_image_upload_options():
    """Display options for selecting between camera and file uploads."""
    st.markdown("📷 Choose an Option for Image Capture" if st.session_state.language == 'English' else "📷 Izaberite opciju za snimanje slike", help = "You can either click a photo using your camera or upload an existing image from your device." if st.session_state.language == 'English' else "Možete ili snimiti fotografiju kamerom ili preneti postojeću sliku sa svog uređaja.")

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
        st.markdown("📸 Use Camera to Capture Image" if st.session_state.language == 'English' else "📸 Koristite kameru za snimanje slike*")
        uploaded_file_camera = st.camera_input("Click to take a photo" if st.session_state.language == 'English' else "Kliknite da uslikate")
        

    if st.session_state.show_uploader:
        st.markdown("📁 Upload Image File" if st.session_state.language == 'English' else "📁 Otpremi sliku*")
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

ms = st.session_state
if "themes" not in ms: 
  ms.themes = {"current_theme": "light",
                    "refreshed": True,
                    
                    "light": {"theme.base": "dark",
                              "theme.backgroundColor": "black",
                              "theme.primaryColor": "#c98bdb",
                              "theme.secondaryBackgroundColor": "#5591f5",
                              "theme.textColor": "#white",
                              "theme.textColor": "#white",
                              "button_face": "🌜",
                              },

                    "dark":  {"theme.base": "light",
                              "theme.backgroundColor": "white",
                              "theme.primaryColor": "#5591f5",
                              "theme.secondaryBackgroundColor": "#82E1D7",
                              "theme.textColor": "#0a1464",
                              "button_face": "🌞"},
                    }
  

def ChangeTheme():
  previous_theme = ms.themes["current_theme"]
  tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
  for vkey, vval in tdict.items(): 
    if vkey.startswith("theme"): st._config.set_option(vkey, vval)

  ms.themes["refreshed"] = False
  if previous_theme == "dark": ms.themes["current_theme"] = "light"
  elif previous_theme == "light": ms.themes["current_theme"] = "dark"
  
# Add custom CSS to style and position the button
st.markdown(
    """
    <style>
    .theme-button {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 9999;
        border: none;
        background-color: #5591f5;
        color: white;
        padding: 10px 20px;
        border-radius: 50%;
        border: none;
        font-size: 18px;
        cursor: pointer;
    }
    .theme-button span {
        margin-left: 0; /* Remove margin to align the icon properly */
    }
    .stButton > button:hover {
        transform: scale(1.1);  /* Slight zoom effect on hover */
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Create the button with custom class for positioning
btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
# Streamlit button to handle theme switching
if st.button(btn_face, key="theme_toggle", on_click=ChangeTheme):
    # This will call the `ChangeTheme` function and refresh the page
    pass

# Ensure the page reloads and refreshes the theme when the button is clicked
if ms.themes["refreshed"] == False:
    ms.themes["refreshed"] = True
    st.rerun()

def main_app():

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
    language_toggle = st.toggle("Prebaci na Srpski")
    st.session_state.language = 'Serbian' if language_toggle else 'English'

    # Sidebar Language Selection
    selected_language = setup_sidebar_language_selection()
    
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
    if uploaded_file_camera or uploaded_file:
        uploaded_file = uploaded_file_camera or uploaded_file

        with st.spinner("Processing image..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                # Perform OCR and translation
                ocr_text, translated_text, detected_language, target_lang = perform_ocr_with_vision_api(
                    temp_path, credentials, selected_language  # Pass the selected language
                )

                # Display OCR and translation results
                st.write(f"**Detected Text ({detected_language.capitalize()}):** {ocr_text}")
                st.write(f"**Translated Text ({target_lang.capitalize()}):** {translated_text}")

            finally:
                os.remove(temp_path)

    # Handle user question input
    if user_question := st.chat_input("😇 Ask Questions" if st.session_state.language == 'English' else "😇 Postavite pitanja"):
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

def main():
    """Main function to render the app."""
    if st.session_state.logged_in:
        main_app()  # Show main app content
    else:
        login()  # Show login page

if __name__ == "__main__":
    main()
