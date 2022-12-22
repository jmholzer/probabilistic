from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from probabilistic.core import (calculate_cdf, calculate_pdf,
                                calculate_quartiles)
from probabilistic.graphics import generate_cdf_figure, generate_pdf_figure
from probabilistic.io import CSVReader


def generate_interface() -> None:
    """Main execution path for generating the app."""
    generate_title()
    generate_body()


def generate_title() -> None:
    """Generate the main title of the app."""
    logo_path = Path("resources/logo.png").resolve()
    st.image(str(logo_path))


def generate_body() -> None:
    """Generate the main tab group and populate with data."""
    st.title("Input", anchor=None)
    generate_input_section()
    st.title("Result")
    generate_results()


def generate_input_section() -> None:
    """Generate content for the input section."""
    st.text_input(label="Security ticker symbol", key="security_ticker")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            label="Current price",
            step=0.1,
            key="current_price",
        )
    with c2:
        st.date_input(label="Estimate price on", key="estimate_date")

    st.session_state["calls"] = pd.DataFrame({"strike": [], "bid": [], "ask": []})
    uploaded_file = st.file_uploader("Call options data", key="1")

    container = st.empty()
    container.dataframe(st.session_state["calls"], width=1024, height=386)
    if uploaded_file is None:
        st.info(f"""Upload a .csv file containing call option data.""")
    else:
        st.session_state["calls"] = pd.read_csv(uploaded_file)
    with container.container():
        st.dataframe(st.session_state["calls"], width=1024, height=386)

    st.multiselect(
        "Optional outputs", ["CDF", "Quartiles"], ["CDF"], key="output_options"
    )


def generate_results() -> None:
    """Generate content for the results section"""
    if not validate_input():
        return

    reader = CSVReader()
    options_data = reader.read(st.session_state["calls"])
    days_forward = _calculate_days_in_future(st.session_state["estimate_date"])

    with st.spinner(text="In progress..."):
        pdf = calculate_pdf(
            options_data, st.session_state["current_price"], days_forward
        )

    if (
        "CDF" in st.session_state["output_options"]
        or "Quartiles" in st.session_state["output_options"]
    ):
        cdf = calculate_cdf(pdf)

    pdf_graph = generate_pdf_figure(
        pdf,
        security_ticker=st.session_state["security_ticker"],
        estimate_date=st.session_state["estimate_date"],
        current_price=st.session_state["current_price"],
    )
    st.pyplot(fig=pdf_graph)

    if "CDF" in st.session_state["output_options"]:
        cdf_graph = generate_cdf_figure(
            cdf,
            security_ticker=st.session_state["security_ticker"],
            estimate_date=st.session_state["estimate_date"],
            current_price=st.session_state["current_price"],
            quartiles=True,
        )
        st.pyplot(fig=cdf_graph)
    if "Quartiles" in st.session_state["output_options"]:
        pass


def validate_input() -> bool:
    """Inspects the app's session_state to check if user input is valid

    Returns:
        True if the current state of the user's input is valid, else False
    """
    return all(
        [
            _validate_security_ticker(),
            _validate_calls(),
            _validate_current_price(),
            _validate_estimate_date(),
        ]
    )


def _validate_security_ticker():
    """Inspects the app's session_state to check if the security ticker is valid

    Returns:
        True if the current state of the user's security name is valid, else False
    """
    # assuming a valid ticker is between 1 and 6 characters long
    result = 1 <= len(st.session_state["security_ticker"]) <= 6
    if not result:
        st.warning("A valid ticker symbol must be specified")
    return result


def _validate_calls():
    """Inspects the app's session_state to check if the user's call option input
    is valid

    Returns:
        True if the current state of the user's call option data input is valid,
        else False
    """
    calls = st.session_state["calls"]
    result = calls.size > 0 and set(calls.columns) == {"strike", "bid", "ask"}
    if not result:
        st.warning("Call options data must be in the specified form")
    return result


def _validate_current_price():
    """Inspects the app's session_state to check if the user's current price input
    is valid

    Returns:
        True if the current state of the user's current price input is valid,
        else False
    """
    result = st.session_state["current_price"] > 0.0
    if not result:
        st.warning("Current price must be greater than 0")
    return result


def _validate_estimate_date():
    """Inspects the app's session_state to check if the user's estimate date input
    is valid

    Returns:
        True if the current state of the user's estimate date input is valid,
        else False
    """
    result = _calculate_days_in_future(st.session_state["estimate_date"]) >= 1
    if not result:
        st.warning("Estimate date must be at least one day in the future")
    return result


def _calculate_days_in_future(input_date: datetime.date) -> int:
    return (input_date - datetime.today().date()).days


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
