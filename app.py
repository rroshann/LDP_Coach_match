import streamlit as st
from pathlib import Path
import runpy

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="LDP Coach Matcher", layout="wide")

# ---------- AUTH SETTINGS ----------
USERNAME_ALLOWED = "ldpteam"   # case-insensitive
PASSWORD_ALLOWED = "LDP@123"

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ---------- SCREEN 1: LOGIN ----------
def show_login():
    st.title("🔐 LDP Coach Matcher - Login")
    st.markdown("Please log in to access the matching system.")
    
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        if username.lower().strip() == USERNAME_ALLOWED.lower() and password == PASSWORD_ALLOWED:
            st.session_state.authenticated = True
            _rerun()
        else:
            st.error("Invalid username or password.")
    
    st.markdown("---")
    st.caption("Contact your LDP administrator if you need access.")

# ---------- SCREEN 2: API KEY ENTRY ----------
def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key by making a minimal test call."""
    if not api_key or not api_key.startswith("sk-"):
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Make a minimal API call to validate
        client.models.list()
        return True
    except Exception:
        return False

def show_api_key_screen():
    st.title("🔑 OpenAI API Key Required")
    st.markdown("""
    This tool uses AI to analyze coach bios and student profiles for better matching.
    
    **You need to provide your own OpenAI API key to continue.**
    
    Don't have a key? [Get one from OpenAI](https://platform.openai.com/api-keys)
    """)
    
    with st.form("api_key_form"):
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            placeholder="sk-proj-..."
        )
        submitted = st.form_submit_button("Validate & Continue", type="primary")
    
    if submitted:
        if not api_key:
            st.error("Please enter an API key.")
        elif not api_key.startswith("sk-"):
            st.error("Invalid key format. OpenAI keys start with 'sk-'.")
        else:
            with st.spinner("Validating API key..."):
                if validate_openai_key(api_key):
                    st.session_state.openai_api_key = api_key
                    st.success("✅ API key validated successfully!")
                    _rerun()
                else:
                    st.error("❌ Invalid API key. Please check and try again.")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Logout"):
            st.session_state.clear()
            _rerun()
    with col2:
        st.caption("Your API key is stored only in your session and never saved to disk.")

# ---------- MAIN FLOW ----------
# Screen 1: Login
if not st.session_state.get("authenticated", False):
    show_login()
    st.stop()

# Screen 2: API Key (after login)
if not st.session_state.get("openai_api_key"):
    show_api_key_screen()
    st.stop()

# Screen 3: Main App (after API key validated)
# Sidebar with logout and key info
with st.sidebar:
    st.success("✅ Logged in as LDPteam")
    st.caption("API Key: ✓ Configured")
    if st.button("Logout"):
        st.session_state.clear()
        _rerun()

# Prevent inner app from calling set_page_config again
_original_set_page_config = st.set_page_config
st.set_page_config = lambda *args, **kwargs: None

# Run the main matching app
target = Path(__file__).with_name("streamlit_app.py")

try:
    runpy.run_path(str(target), run_name="__main__")
finally:
    st.set_page_config = _original_set_page_config
