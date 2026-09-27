import os
import sys
from odf.opendocument import load
from odf.text import ListItem
from odf.teletype import extractText
import streamlit as st
from odf import text
from groq import Groq

st.set_page_config(page_title="Patient Assistant", page_icon=" ",layout='wide')
st.title("Patient Assistant")
##############################################################################
def load_sys_part(file):
    """Load and extract all text content from an .odt file, including
    text nested inside formatted spans (bold, italic, colored, etc.)."""
    document = load(file)
    all_text = []
    for element in document.getElementsByType(text.P):
        all_text.append(extractText(element))
    return "\n".join(all_text)

def build_groq_messages(messages, infos):
    """Build OpenAI/Groq-style messages list (role/content strings),
    folding reference info into the first user turn."""
    built = []
    for i, msg in enumerate(messages):
        content = msg["content"]
        if i == 0:
            content = "Informations disponibles : " + infos + "\n\nQuestion : " + content
        built.append({"role": msg["role"], "content": content})
    return built
import markdown as md

def render_message(role, content):
    html_content = md.markdown(content, extensions=["extra", "nl2br"])
    if role == "user":
        st.html(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                <div style="background-color: #0b57d0; color: white; padding: 10px 16px; border-radius: 18px 18px 2px 18px; max-width: 70%;">
                    {html_content}
                </div>
            </div>
        """)
    else:
        st.html(f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                <div style="background-color: #f0f4f9; color: #1f1f1f; padding: 10px 16px; border-radius: 18px 18px 18px 2px; max-width: 70%;">
                    {html_content}
                </div>
            </div>
        """)

def run_pipeline(system_prompt, messages):
    client = Groq(api_key="")

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=full_messages
    )
    return response.choices[0].message.content
#############################################################################################
col1,col2=st.columns([1,4])
with col1:
  with st.container(height=80,border=True):
      user_input=st.chat_input("Ask your question")
with col2:
    with st.container(height=700,border=True):
        system_prompt=load_sys_part('/system_prompt.odt')
        knowledge_base=load_sys_part('/knowledge_base.odt')
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                render_message(message["role"],message["content"])
        if user_input:
            with st.chat_message("user"):
                render_message("user",user_input)
            USER_INPUT='\n'.join(['<question>',user_input,'</question>'])
            st.session_state.messages.append({"role": "user", "content": USER_INPUT})
            with st.chat_message("assistant"):
                with st.spinner("Réflexion en cours..."):
                    response_text = run_pipeline(
                            system_prompt=system_prompt, 
                            messages=build_groq_messages(st.session_state.messages,knowledge_base)
                    )
                    render_message("assistant",response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})             
    ######################################
