from pathlib import Path

import pandas as pd
import streamlit as st

from oipd.core import calculate_cdf, calculate_pdf
from oipd.graphics import generate_cdf_figure, generate_pdf_figure
from oipd.io import CSVReader


def generate_interface() -> None:
    """Main execution path for generating the app."""
    _set_page_upper_margin()
    generate_title()
    generate_body()


def generate_title() -> None:
    """Generate the main title of the app."""
    interface_path = Path(__file__).parent.resolve()
    logo_path = interface_path / Path("resources/logo.png")
    st.image(str(logo_path), use_column_width=True)


def generate_body() -> None:
    """Generate the main tab group and populate with data."""
    st.title("Input", anchor=None)
    generate_input_section()
    st.title("Result")
    generate_results()


def generate_input_section() -> None:
    """Generate content for the input section."""
    st.text_input(label="Security ticker symbol", key="security_ticker")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input(
            label="Current price",
            step=0.1,
            key="current_price",
        )
    with c2:
        st.date_input(label="Current date", key="current_date")
    with c3:
        st.date_input(label="Option expires on", key="expiry_date")

    st.session_state["calls"] = pd.DataFrame({"strike": [], "last_price": []})
    uploaded_file = st.file_uploader("Call options data", key="1")

    container = st.empty()
    container.dataframe(st.session_state["calls"], width=1024, height=386)
    if uploaded_file is None:
        st.info("""Upload a .csv file containing call option data.""")
    else:
        st.session_state["calls"] = pd.read_csv(uploaded_file)
    with container.container():
        st.dataframe(st.session_state["calls"], width=1024, height=386)

    st.multiselect("Optional outputs", ["CDF"], ["CDF"], key="output_options")


def generate_results() -> None:
    """Generate content for the results section"""
    if not validate_input():
        return

    reader = CSVReader()
    options_data = reader.read(st.session_state["calls"])
    days_forward = _calculate_days_in_future(
        st.session_state["expiry_date"], st.session_state["current_date"]
    )

    with st.spinner(text="Calculating..."):
        pdf = calculate_pdf(
            options_data, st.session_state["current_price"], days_forward
        )

    pdf_graph = generate_pdf_figure(
        pdf,
        security_ticker=st.session_state["security_ticker"],
        expiry_date=st.session_state["expiry_date"],
        current_price=st.session_state["current_price"],
    )
    st.subheader("PDF")
    st.pyplot(fig=pdf_graph)
    st.markdown("""---""")

    if "CDF" in st.session_state["output_options"]:
        with st.spinner(text="Calculating..."):
            cdf = calculate_cdf(pdf)
        cdf_graph = generate_cdf_figure(
            cdf,
            security_ticker=st.session_state["security_ticker"],
            expiry_date=st.session_state["expiry_date"],
            current_price=st.session_state["current_price"],
            quartiles=True,
        )
        st.subheader("CDF")
        st.pyplot(fig=cdf_graph)
        st.markdown("""---""")


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
            _validate_expiry_date(),
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
    result = calls.size > 0 and set(calls.columns) == {
        "strike",
        "last_price",
    }
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


def _validate_expiry_date():
    """Inspects the app's session_state to check if the user's estimate date input
    is valid

    Returns:
        True if the current state of the user's estimate date input is valid,
        else False
    """
    result = (
        _calculate_days_in_future(
            st.session_state["expiry_date"], st.session_state["current_date"]
        )
        >= 1
    )
    if not result:
        st.warning("Expiry date must be at least one day in the future")
    return result


def _calculate_days_in_future(future_date, curr_date) -> int:
    return (future_date - curr_date).days


def _set_page_upper_margin():
    """Set the upper margin of the page to 0"""
    st.markdown(
        f"""
            <style>
                .appview-container .main .block-container{{
                    padding-top: {3}rem;
                }}
            </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    generate_interface()
