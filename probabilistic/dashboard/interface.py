import streamlit as st
import pandas as pd


def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


options_chain = pd.DataFrame({"strike": [], "bid": [], "ask": []})
st.title("Probabilistic")
uploaded_file = st.file_uploader("", key="1")
container = st.empty()
container.table(options_chain)
if uploaded_file is None:
    st.info(
        f"""
                Upload a .csv file containing options data.
                """
    )
    st.stop()
else:
    file_container = st.expander("Check your uploaded .csv")
    options_chain = pd.read_csv(uploaded_file)

with container.container():
    st.table(options_chain)