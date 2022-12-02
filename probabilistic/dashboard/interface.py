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


st.title("Probabilistic")
uploaded_file = st.file_uploader(
    "",
    key="1"
)
if uploaded_file is not None:
    file_container = st.expander("Check your uploaded .csv")
    shows = pd.read_csv(uploaded_file)
else:
    st.info(
        f"""
            Upload a .csv file containing options data.
            """
    )

    st.stop()
st.table(shows)
