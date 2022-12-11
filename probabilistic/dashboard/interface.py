from pathlib import Path

import pandas as pd
import streamlit as st

from typing import Optional, Dict


def generate_interface() -> None:
    """Main execution path for generating the app.
    """
    generate_title()
    generate_body()


def generate_title() -> None:
    """Generate the main title of the app.
    """
    logo_path = Path("resources/logo.png").resolve()
    st.image(str(logo_path))


def generate_body() -> None:
    """Generate the main tab group and populate with data.
    """
    st.title("Security Details", anchor=None)
    generate_input_section()
    st.title("Result")
    generate_results()


def generate_input_section() -> None:
    """Generate content for the input section.
    """
    security_name = st.text_input("Security name")
    c1, c2 = st.columns(2)
    with c1:
        current_price = st.number_input('Current price', step=0.1)
    with c2:
        estimate_date = st.date_input('Estimate price on')

    options_chain = pd.DataFrame({"strike": [], "bid": [], "ask": []})
    uploaded_file = st.file_uploader("Call options data", key="1")

    container = st.empty()
    container.dataframe(options_chain, width=1024, height=386)
    if uploaded_file is None:
        st.info(f"""Upload a .csv file containing call option data.""")
    else:
        options_chain = pd.read_csv(uploaded_file)
    with container.container():
        st.dataframe(options_chain, width=1024, height=386)

    output_options = st.multiselect(
        "Show me the following ",
        ["PDF", "CDF"],
        ["PDF", "CDF"]
    )


def generate_results(state: Optional[Dict] = None) -> None:
    """Generate content for the results section
    """
    if not state:
        st.info("Enter valid input to generate results")


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


if __name__ == "__main__":
    generate_interface()
