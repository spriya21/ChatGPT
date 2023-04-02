import streamlit as st
import csv

from model import Bot

RESPONSE_FILE = "response.csv"

def blank(): return st.text('')

st.subheader("FIFA World Cup Chatbot ⚽️")
blank()


query = st.text_area("Ask me anything world cup related...")
blank()

search = st.button("search")
if 'res' not in st.session_state:
    st.session_state["res"] = None
if 'search' not in st.session_state or not st.session_state["search"]:
    st.session_state["search"] = search

if search:

    with st.spinner("Generating Response 🤖"):
        ai = Bot()
        res = ai.generate_response(query=query)
        st.session_state['res'] = res

if st.session_state["res"]:
    blank()
    st.markdown("###### Please, select the answers that you would like to vote for.")
    #oai = st.checkbox(f"OpenAI: {st.session_state['res']['oai']}")
    ops = st.checkbox(f"OpenSource: {st.session_state['res']['ops']}")
    voted = st.button("Vote!")

    if voted:
        st.success('Thank you for voting! Your response has been saved.', icon="✅")
        with open(RESPONSE_FILE, 'a', newline='') as csvfile:
            fieldnames = [
                #'Question', 'OpenAI Answer', 'OpenAI Vote',
                'Opensource Answer', 'Opensource Vote'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({
                'Question': query,
                #'OpenAI Answer': st.session_state['res']['oai'],
                #'OpenAI Vote': int(oai),
                'Opensource Answer': st.session_state['res']['ops'],
                'Opensource Vote': int(ops)
            })
