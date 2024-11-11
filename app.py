import os
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import dotenv
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

dotenv.load_dotenv(dotenv.find_dotenv())

st.set_page_config(page_title="Serbian GPT", page_icon="💫")

# Initialize Hugging Face VQA model and processor
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model1' not in st.session_state:
        st.session_state.model1 = 'llama3-70b-8192'
    if 'model2' not in st.session_state:
        st.session_state.model2 = 'gemma-7b-it'
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = 'Chatbot'
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

def display_customization_options():
    st.sidebar.title('Customization' if st.session_state.language == 'English' else 'Prilagođavanje')
    chat_mode_label = 'Select Mode' if st.session_state.language == 'English' else 'Izaberite režim'
    chat_mode = st.sidebar.selectbox(chat_mode_label, ['Chatbot', 'Chat with Image'] if st.session_state.language == 'English' else ['Razgovor sa AI', 'Razgovor sa slikom'])
    conversational_memory_label = 'Conversational memory length' if st.session_state.language == 'English' else 'Dužina memorije razgovora'
    conversational_memory_length = st.sidebar.slider(conversational_memory_label, 1, 10, value=5)
    return chat_mode, conversational_memory_length

def initialize_groq_chat(groq_api_key, model_name):
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

def initialize_conversation(groq_chat, memory):
    return ConversationChain(llm=groq_chat, memory=memory)

def analyze_image_with_vqa(uploaded_image, user_question, max_new_tokens=50):
    inputs = vqa_processor(uploaded_image, user_question, return_tensors="pt")
    vqa_response = vqa_model.generate(**inputs, max_new_tokens=max_new_tokens)
    response_text = vqa_processor.decode(vqa_response[0], skip_special_tokens=True)
    return response_text

def process_user_question(user_question, conversation1, conversation2, conversational_memory_length, uploaded_image=None):
    user_question_for_model = user_question
    if st.session_state.language == 'Serbian':
        user_question_for_model = "Molim vas, odgovarajte na srpskom: " + user_question

    if uploaded_image:
        image_response = analyze_image_with_vqa(uploaded_image, user_question_for_model)
        user_question_for_model += f" (Анализа слике: {image_response})"

    response1 = conversation1(user_question_for_model).get('response', '').strip()
    response2 = conversation2(user_question_for_model).get('response', '').strip()

    hybrid_response = response1 if response1 == response2 else f"{response1} {response2}"
    if conversational_memory_length > 5:
        hybrid_response += " (Based on extended memory context)"

    if not st.session_state.chat_history or st.session_state.chat_history[-1]['human'] != user_question:
        st.session_state.chat_history.append({'human': user_question, 'AI': hybrid_response})
    else:
        st.session_state.chat_history[-1]['AI'] = hybrid_response

def display_chat_history():
    st.sidebar.subheader("Chat History" if st.session_state.language == 'English' else "Istorija razgovora")
    for message in st.session_state.chat_history:
        st.sidebar.markdown(f"🧑 **You:** {message['human']}")
        if message['AI']:
            st.sidebar.markdown(f"🤖 **AI:** {message['AI']}\n")

def main():
    groq_api_key = os.environ['GROQ_API_KEY']
    initialize_session_state()

    language_toggle = st.toggle("Switch to Serbian")
    st.session_state.language = 'Serbian' if language_toggle else 'English'
    
    title_text = "Serbia-GPT 💫" if st.session_state.language == 'English' else "Srbija-GPT 💫"
    st.title(title_text)
    welcome_text = "Chat with Serbia GPT, an ultra-fast AI chatbot!" if st.session_state.language == 'English' else "Razgovarajte sa Srbija GPT, izuzetno brzim AI četbotom!"
    st.markdown(welcome_text)

    chat_mode, conversational_memory_length = display_customization_options()
    st.session_state.chat_mode = chat_mode

    memory = ConversationBufferWindowMemory(k=conversational_memory_length)
    display_chat_history()
    st.divider()

    if chat_mode == 'Chat with Image' or chat_mode == 'Razgovor sa slikom':
        uploaded_file = st.file_uploader("Upload an image" if st.session_state.language == 'English' else "Otpremi sliku", type=["jpeg", "jpg", "png"])
        if uploaded_file is not None:
            with st.spinner("Uploading file..." if st.session_state.language == 'English' else "Otpremanje datoteke..."):
                st.session_state.uploaded_image = Image.open(uploaded_file)
                st.success(f"File {uploaded_file.name} uploaded successfully!" if st.session_state.language == 'English' else f"Datoteka {uploaded_file.name} je uspešno otpremljena!")

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
                conversational_memory_length,
                uploaded_image=st.session_state.uploaded_image
            )

        ai_response = st.session_state.chat_history[-1]["AI"]
        with st.chat_message("assistant"):
            st.markdown(ai_response)

if __name__ == "__main__":
    main()
