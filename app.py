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

# Ensure consistent language detection
DetectorFactory.seed = 0

# Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv())

# Streamlit page settings
st.set_page_config(page_title="Srpski AI", page_icon="📸")
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Show a login form using an expander (simulating a popup)
def login_popup():
    """Show the login form as a simulated popup."""
    with st.expander("Login to Srpski AI ✨", expanded=True):  # Expander simulates a modal-like effect
        username = st.text_input("Username 👤")
        password = st.text_input("Password 🔑", type="password")
        if st.button("Login ⚡"):
            if username == "user_name" and password == "user@123":
                st.session_state.logged_in = True
                st.success("Valid Credentials ✅")
                st.success("Click the login button once more to login! 😇")
            else:
                st.error("Invalid username or password")


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
    st.sidebar.title("Translate text in Image" if st.session_state.language == 'English' else "Prevedi tekst na slici", help = "Select the language for Image Translation" if st.session_state.language == 'English' else "Izaberite jezik za prevođenje slike")
    #st.sidebar.info("Select the language for translation.")

    # Create an instance of GoogleTranslator to access supported languages
    translator = GoogleTranslator()
    available_languages = translator.get_supported_languages(as_dict=False)
    
    # Add a dropdown for language selection
    translation_language = st.sidebar.selectbox(
        "Select a Language" if st.session_state.language == 'English' else "Izaberite jezik", 
        options=available_languages, 
        index=available_languages.index("english")  # Default to English
    )

    # Save selected language in session state
    st.session_state.selected_translation_language = translation_language
    return translation_language

#is this needs to be edited
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

def initialize_language_options():
    """Initialize the sidebar for selecting interaction language."""
    if "chat_language" not in st.session_state:
        st.session_state.chat_language = "en"

    st.sidebar.title("Chat Language Options" if st.session_state.language == 'English' else "Opcije jezika za chat", help = "Select the language in which the chatbot should respond!" if st.session_state.language == 'English' else "Izaberite jezik na kojem chatbot treba da odgovara!")
    
    # Full list of languages and their codes
    available_languages = {
        "Afrikaans": "af",
        "Albanian": "sq",
        "Amharic": "am",
        "Arabic": "ar",
        "Armenian": "hy",
        "Assamese": "as",
        "Aymara": "ay",
        "Azerbaijani": "az",
        "Bambara": "bm",
        "Basque": "eu",
        "Belarusian": "be",
        "Bengali": "bn",
        "Bhojpuri": "bho",
        "Bosnian": "bs",
        "Bulgarian": "bg",
        "Catalan": "ca",
        "Cebuano": "ceb",
        "Chichewa": "ny",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Corsican": "co",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dhivehi": "dv",
        "Dogri": "doi",
        "Dutch": "nl",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Ewe": "ee",
        "Filipino": "tl",
        "Finnish": "fi",
        "French": "fr",
        "Frisian": "fy",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greek": "el",
        "Guarani": "gn",
        "Gujarati": "gu",
        "Haitian Creole": "ht",
        "Hausa": "ha",
        "Hawaiian": "haw",
        "Hebrew": "iw",
        "Hindi": "hi",
        "Hmong": "hmn",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Igbo": "ig",
        "Ilocano": "ilo",
        "Indonesian": "id",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jw",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Khmer": "km",
        "Kinyarwanda": "rw",
        "Konkani": "gom",
        "Korean": "ko",
        "Krio": "kri",
        "Kurdish (Kurmanji)": "ku",
        "Kurdish (Sorani)": "ckb",
        "Kyrgyz": "ky",
        "Lao": "lo",
        "Latin": "la",
        "Latvian": "lv",
        "Lingala": "ln",
        "Lithuanian": "lt",
        "Luganda": "lg",
        "Luxembourgish": "lb",
        "Macedonian": "mk",
        "Maithili": "mai",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malayalam": "ml",
        "Maltese": "mt",
        "Maori": "mi",
        "Marathi": "mr",
        "Meiteilon (Manipuri)": "mni-Mtei",
        "Mizo": "lus",
        "Mongolian": "mn",
        "Myanmar": "my",
        "Nepali": "ne",
        "Norwegian": "no",
        "Odia (Oriya)": "or",
        "Oromo": "om",
        "Pashto": "ps",
        "Persian": "fa",
        "Polish": "pl",
        "Portuguese": "pt",
        "Punjabi": "pa",
        "Quechua": "qu",
        "Romanian": "ro",
        "Russian": "ru",
        "Samoan": "sm",
        "Sanskrit": "sa",
        "Scots Gaelic": "gd",
        "Sepedi": "nso",
        "Serbian": "sr",
        "Sesotho": "st",
        "Shona": "sn",
        "Sindhi": "sd",
        "Sinhala": "si",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Somali": "so",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swedish": "sv",
        "Tajik": "tg",
        "Tamil": "ta",
        "Tatar": "tt",
        "Telugu": "te",
        "Thai": "th",
        "Tigrinya": "ti",
        "Tsonga": "ts",
        "Turkish": "tr",
        "Turkmen": "tk",
        "Twi": "ak",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uyghur": "ug",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Xhosa": "xh",
        "Yiddish": "yi",
        "Yoruba": "yo",
        "Zulu": "zu"
    }

    # Display dropdown for language selection
    selected_language = st.sidebar.selectbox(
        "Select a Language" if st.session_state.language == 'English' else "Izaberite jezik",
        options=list(available_languages.keys()),
        index=list(available_languages.keys()).index("English"),
    )

    # Save selected chatbot language code in session state
    st.session_state.chatbot_language = available_languages[selected_language]
    return selected_language

def process_user_question(user_question, conversation_model, chatbot_language):
    """
    Process the user question and generate a chatbot response in the selected language.
    """
    # Initialize chat history in session state if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Step 1: Translate user input to English for processing
    if chatbot_language != "en":  # Use language code "en" for English
        try:
            user_question_translated = GoogleTranslator(source="auto", target="en").translate(user_question)
        except Exception as e:
            st.error(f"Error translating user input to English: {e}")
            return
    else:
        user_question_translated = user_question

    # Step 2: Generate response using the selected model
    try:
        response = conversation_model(user_question_translated).get("response", "").strip()
    except Exception as e:
        st.error(f"Error generating chatbot response: {e}")
        return

    # Step 3: Translate the response back to the selected chatbot language
    if chatbot_language != "en":
        try:
            response_translated = GoogleTranslator(source="en", target=chatbot_language).translate(response)
        except Exception as e:
            st.error(f"Error translating AI response to {chatbot_language}: {e}")
            response_translated = response  # Fallback to untranslated response
    else:
        response_translated = response

    # Step 4: Append the interaction to the chat history if not already added
    new_entry = {"human": user_question, "AI": response_translated}
    if not st.session_state.chat_history or st.session_state.chat_history[-1] != new_entry:
        st.session_state.chat_history.append(new_entry)

    # Step 5: Display the assistant response (No need to re-render user question here)
    with st.chat_message("assistant"):
        st.markdown(response_translated)




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

ms = st.session_state
if "themes" not in ms: 
  ms.themes = {"current_theme": "light",
                    "refreshed": True,
                    
                    "light": {"theme.base": "dark",
                              "theme.backgroundColor": "black",
                              "theme.primaryColor": "#c98bdb",
                              "theme.secondaryBackgroundColor": "#5591f5",
                              "theme.textColor": "white",
                              "theme.textColor": "white",
                              "button_face": "🌜"},

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

btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
st.button(btn_face, on_click=ChangeTheme)

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

    st.markdown("👆 Open the sidebar to select language")
    # Streamlit UI for language toggle
    language_toggle = st.toggle("Prebaci na Srpski")
    st.session_state.language = 'Serbian' if language_toggle else 'English'

    # Sidebar Language Selection
    translation_language = setup_sidebar_language_selection()

    # Sidebar Language Selection
    chatbot_language = initialize_language_options()
    
    # Set page title and instructions
    title_text = "Srpski AI 📸"
    st.title(title_text)
    welcome_text = (
        "Translate text from photos and chat with Srpski AI in any language!"
        if st.session_state.language == 'English'
        else "Prevedite tekst sa fotografija i razgovarajte sa Srpski AI na bilo kom jeziku!"
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
                    temp_path, credentials, translation_language  # Pass the selected language
                )

                # Display OCR and translation results
                st.write(f"**Detected Text ({detected_language.capitalize()}):** {ocr_text}")
                st.write(f"**Translated Text ({target_lang.capitalize()}):** {translated_text}")

            finally:
                os.remove(temp_path)

    if user_question := st.chat_input("Ask anything ✨"):
        with st.chat_message("user"):
            st.markdown(user_question)  # Render user question here

        # Initialize chatbot model
        groq_chat_model = initialize_groq_chat(os.environ["GROQ_API_KEY"], st.session_state.model1)
        conversation_model = initialize_conversation(groq_chat_model, memory)

        # Process the user question
        with st.spinner("The bot is thinking..."):
            process_user_question(user_question, conversation_model, st.session_state.chatbot_language)

# Main function to handle page rendering
def main():
    """Main function to handle page rendering."""
    
    if not st.session_state.logged_in:
        login_popup()  # Show the login form if the user is not logged in
    else:
        main_app()  # Render the main app content

if __name__ == "__main__":
    main()
