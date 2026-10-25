import base64
import streamlit as st

def apply_css():
    st.markdown(
        """
        <style>
        h1 { color: black !important; }
        [data-testid="stSubheader"] { color: black !important; }
        label { color: black !important; font-weight: 600; }
        input, textarea {
            color: black !important;
            background-color: rgba(255,255,255,0.85) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def set_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
