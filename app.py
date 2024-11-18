import os
import json
import requests
import streamlit as st
from google.cloud import vision
import dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Ensure consistent language detection
DetectorFactory.seed = 0

# Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv())

# Streamlit page settings
st.set_page_config(page_title="Serbian GPT", page_icon="💫")

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "D:/IMP- DATA/Download/gentle-impulse-442016-m5-8c9a87a4f3a8.json"

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

def initialize_conversation(groq_chat, memory):
    """Initialize conversation chain with memory."""
    return ConversationChain(llm=groq_chat, memory=memory)

def perform_ocr_with_vision_api(image_path):
    """Performs OCR using Google Cloud Vision API and translates detected text."""
    client = vision.ImageAnnotatorClient()
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
            # Fallback to langdetect
            detected_language = detect(ocr_text)
        except Exception as e:
            print(f"Language detection failed: {e}")
            detected_language = "unknown"
    
    try:
        # Handle translation
        if detected_language in ['sr', 'mk', 'unknown']:  # Assume Serbian/Macedonian for Cyrillic
            translated_text = GoogleTranslator(source="auto", target="en").translate(ocr_text)
            target_language = "English"
        elif detected_language == 'en':  # English detected
            translated_text = GoogleTranslator(source="en", target="sr").translate(ocr_text)
            target_language = "Serbian"
        else:
            translated_text = f"Text detected as {detected_language}, which is not supported for translation."
            target_language = "Unknown"
    except Exception as e:
        translated_text = f"Translation failed. Extracted text: {ocr_text}"
        target_language = "Unknown"
        print(f"Error during translation: {e}")

    return ocr_text, translated_text, detected_language, target_language


def process_user_question(user_question, conversation1, conversation2, uploaded_image=None, ocr_text=""):
    """Processes the user question and generates a hybrid response."""
    user_question_for_model = user_question
    if st.session_state.language == 'Serbian':
        user_question_for_model = "Molim vas, odgovarajte na srpskom: " + user_question

    if uploaded_image and ocr_text:
        # Use the OCR translated text as the bot's response
        user_question_for_model += f" (Tekst iz slike: {ocr_text})"

    response1 = conversation1(user_question_for_model).get('response', '').strip()
    response2 = conversation2(user_question_for_model).get('response', '').strip()

    # Merge the two responses into a hybrid response
    hybrid_response = response1 if response1 == response2 else f"{response1} {response2}"

    # Add conversation to history
    if not st.session_state.chat_history or st.session_state.chat_history[-1]['human'] != user_question:
        st.session_state.chat_history.append({'human': user_question, 'AI': hybrid_response})
    else:
        st.session_state.chat_history[-1]['AI'] = hybrid_response

def display_chat_history():
    """Displays the chat history in the sidebar.""" 
    st.sidebar.subheader("Chat History" if st.session_state.language == 'English' else "Istorija razgovora")
    for message in st.session_state.chat_history:
        st.sidebar.markdown(f"🧑 *You:* {message['human']}")
        if message['AI']:
            st.sidebar.markdown(f"🤖 *AI:* {message['AI']}\n")

def main():
    groq_api_key = os.environ['GROQ_API_KEY']
    initialize_session_state()

    # Language toggle (button for switching between English and Serbian)
    language_toggle = st.toggle("Switch to Serbian")
    st.session_state.language = 'Serbian' if language_toggle else 'English'

    title_text = "Serbia-GPT 💫" if st.session_state.language == 'English' else "Srbija-GPT 💫"
    st.title(title_text)
    welcome_text = "Chat with Serbia GPT, an ultra-fast AI chatbot!" if st.session_state.language == 'English' else "Razgovarajte sa Srbija GPT, izuzetno brzim AI četbotom!"
    st.markdown(welcome_text)

    memory = ConversationBufferWindowMemory(k=10)
    display_chat_history()
    st.divider()

    uploaded_file = st.file_uploader("Upload an image" if st.session_state.language == 'English' else "Otpremi sliku", type=["jpeg", "jpg", "png"])
    if uploaded_file is not None:
        with st.spinner("Processing image for text..." if st.session_state.language == 'English' else "Obrada slike za tekst..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                # Perform OCR and translation
                ocr_text, translated_text, detected_language, target_language = perform_ocr_with_vision_api(temp_path)
                
                # Save translated text to session state
                st.session_state.ocr_text = translated_text
                
                st.success(
                    f"Text detected in {detected_language.capitalize()} and translated to {target_language.capitalize()}!"
                    if st.session_state.language == 'English'
                    else f"Tekst prepoznat na {detected_language.capitalize()} i preveden na {target_language.capitalize()}!"
                )
                
                # Display as a bot response
                st.markdown(f"**Extracted Text ({detected_language.capitalize()}):** {ocr_text}\n\n**Translated to {target_language.capitalize()}:** {translated_text}")
            finally:
                os.remove(temp_path)

    if user_question := st.chat_input("Ask Questions" if st.session_state.language == 'English' else "Postavite pitanja"):
        if not st.session_state.chat_history or st.session_state.chat_history[-1]["human"] != user_question:
            st.session_state.chat_history.append({"human": user_question, "AI": ""})
        with st.chat_message("user"):
            st.markdown(user_question)

        groq_chat_model1 = initialize_groq_chat(groq_api_key, st.session_state.model1)
        groq_chat_model2 = initialize_groq_chat(groq_api_key, st.session_state.model2)
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
