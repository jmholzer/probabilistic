import streamlit as st
import pandas as pd


def generate_interface() -> None:
    """Main execution path for generating the app.
    """
    generate_title()
    generate_tab_group()


def generate_title() -> None:
    """Generate the main title of the app.
    """
    st.title("Probabilistic")


def generate_tab_group() -> None:
    """Generate the main tab group and populate with data.
    """
    input_tab, results_tab = st.tabs(["Input", "Results"])
    with input_tab:
        generate_input_tab()
    with results_tab:
        st.write("blank")


def generate_input_tab() -> None:
    """Generate content for the input page.
    """
    options_chain = pd.DataFrame({"strike": [], "bid": [], "ask": []})
    uploaded_file = st.file_uploader("", key="1")
    container = st.empty()
    container.dataframe(options_chain, width=1024, height=386)

    if uploaded_file is None:
        st.info(f"""Upload a .csv file containing options data.""")
    else:
        options_chain = pd.read_csv(uploaded_file)

    with container.container():
        st.dataframe(options_chain, width=1024, height=386)


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
