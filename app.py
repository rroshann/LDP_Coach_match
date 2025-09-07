import runpy
import streamlit as st
from pathlib import Path

# ---------- PAGE CONFIG (call exactly once, before any other Streamlit calls) ----------
st.set_page_config(page_title="Login", layout="centered")

# ---------- AUTH SETTINGS ----------
USERNAME_ALLOWED = "ldpteam"   # case-insensitive
PASSWORD_ALLOWED = "LDP@123"

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def show_login():
    st.title("🔐 Login Required")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if u.lower().strip() == USERNAME_ALLOWED and p == PASSWORD_ALLOWED:
            st.session_state.authenticated = True
            _rerun()
        else:
            st.error("Invalid username or password.")

# ---------- LOGIN GATE ----------
if not st.session_state.get("authenticated", False):
    show_login()
    st.stop()

# ---------- POST-LOGIN SHELL ----------
# Sidebar with logout
with st.sidebar:
    st.success("Logged in as LDPteam")
    if st.button("Logout"):
        st.session_state.clear()
        _rerun()

# We already set page config above; prevent your inner app from calling it again.
# Monkey-patch st.set_page_config to a no-op for the duration of the run.
_original_set_page_config = st.set_page_config
st.set_page_config = lambda *args, **kwargs: None

# Optional: set a friendlier title after login (safe to use st.* now)
st.title("Launching app…")

# ---------- RUN YOUR EXISTING FILE WITHOUT INLINING ----------
# Keep your file name the same and in the same folder.
target = Path(__file__).with_name("streamlit_app_v8.py")

try:
    runpy.run_path(str(target), run_name="__main__")
finally:
    # restore in case of hot-reload
    st.set_page_config = _original_set_page_config
