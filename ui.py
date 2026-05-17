import streamlit as st

from llamaindex import query_rag

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Enterprise RAG Chat",
    layout="wide",
)

st.title(
    "Enterprise RAG Chat"
)

# =========================
# SESSION STATE
# =========================

if "history" not in (
    st.session_state
):

    st.session_state.history = []

# =========================
# INPUT FORM
# =========================

with st.form("chat_form"):

    user_input = st.text_input(
        "Ask me anything:"
    )

    submitted = (
        st.form_submit_button(
            "Submit"
        )
    )

# =========================
# QUERY
# =========================

if (
    submitted
    and user_input.strip()
):

    with st.spinner(
        "Thinking..."
    ):

        try:

            response, sources = (
                query_rag(
                    user_input
                )
            )

            (
                st.session_state
                .history
                .append({
                    "question":
                    user_input,

                    "response":
                    response,

                    "sources":
                    sources,
                })
            )

        except Exception as e:

            st.error(str(e))

# =========================
# HISTORY
# =========================

for chat in reversed(
    st.session_state.history
):

    st.markdown("### You")

    st.write(
        chat["question"]
    )

    st.markdown("### RAG")

    st.write(
        chat["response"]
    )

    with st.expander(
        "Sources"
    ):

        st.json(
            chat["sources"]
        )