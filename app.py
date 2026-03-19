"""
app.py — Streamlit chatbot UI
"""

import streamlit as st
from rag_pipeline import ask, load_qa_chain

# Page config 
st.set_page_config(
    page_title="SG Employment Rights Advisor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.82em;
        color: #a0a8c0;
    }
    .blocked-badge {
        background: #3d1a1a;
        border: 1px solid #7a2a2a;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.78em;
        color: #f08080;
        margin-bottom: 8px;
        display: inline-block;
    }
    .confidence-high {
        color: #4ade80;
        font-size: 0.75em;
        font-family: monospace;
    }
    .confidence-low {
        color: #f59e0b;
        font-size: 0.75em;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

#  Sidebar 
with st.sidebar:
    st.title("SG Employment Advisor")
    st.caption("Powered by official MOM & CPF sources")

    st.divider()

    st.subheader("💡 Try asking:")
    example_questions = [
        "What is the minimum notice period if I resign?",
        "How many days of annual leave am I entitled to?",
        "Can my employer deduct money from my salary?",
        "What is the overtime pay rate in Singapore?",
        "Am I entitled to sick leave?",
        "How much CPF does my employer contribute?",
        "What should I do if I'm wrongfully dismissed?",
        "What are the rules for maternity leave?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending_question = q

    st.divider()

    st.subheader("Need more help?")
    st.markdown("""
    - **MOM Helpline:** 6438 5122
    - **TADM (Disputes):** 1800 221 9088
    - **ECT (Tribunals):** employmentclaims.gov.sg
    """)

    st.divider()
    st.caption(
        " This chatbot provides general information only. "
        "It is not legal advice. Always verify with MOM or consult a lawyer "
        "for your specific situation."
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

# Main chat 
st.title("Singapore Employment Rights Advisor")
st.caption("Ask about your workplace rights — answers sourced from official MOM, CPF, and Tripartite guidelines.")

# Warm up the pipeline (loads models into cache)
with st.spinner("Loading AI models... (first load takes ~30 seconds)"):
    try:
        load_qa_chain()
        st.success(" Ready", icon="✅")
    except FileNotFoundError as e:
        st.error(f" {e}")
        st.stop()
    except ValueError as e:
        st.error(f" {e}")
        st.stop()

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("blocked"):
            st.markdown('<span class="blocked-badge"> Out of scope — redirected</span>', unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) used", expanded=False):
                for src in msg["sources"]:
                    source_name = src.metadata.get("source", "Unknown source")
                    source_url = src.metadata.get("url", "")
                    preview = src.page_content[:220].replace("\n", " ")
                    link = f' — <a href="{source_url}" target="_blank">↗</a>' if source_url else ""
                    st.markdown(
                        f'<div class="source-card"><strong>{source_name}</strong>{link}<br>{preview}...</div>',
                        unsafe_allow_html=True
                    )

            # Confidence indicator
            n_sources = len(msg["sources"])
            if n_sources >= 3:
                st.markdown('<span class="confidence-high">● High confidence — well-supported by sources</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="confidence-low">● Moderate confidence — limited source coverage</span>', unsafe_allow_html=True)

# Handle sidebar button clicks
if st.session_state.pending_question:
    query = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    query = st.chat_input("Ask about your Singapore employment rights...")

# Process query
if query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching official sources..."):
            answer, sources, was_blocked, query_trace = ask(query)

        # Show rewrite banner if query was normalised
        if query_trace.get("was_rewritten"):
            st.info(
                f"🔍 Interpreted as: *\"{query_trace['final']}\"*",
                icon="✏️"
            )

        st.markdown(answer)

        if was_blocked:
            st.markdown('<span class="blocked-badge"> Out of scope — redirected</span>', unsafe_allow_html=True)

        if sources:
            with st.expander(f"📚 {len(sources)} source(s) used", expanded=False):
                for src in sources:
                    source_name = src.metadata.get("source", "Unknown source")
                    source_url = src.metadata.get("url", "")
                    preview = src.page_content[:220].replace("\n", " ")
                    link = f' — <a href="{source_url}" target="_blank">↗</a>' if source_url else ""
                    st.markdown(
                        f'<div class="source-card"><strong>{source_name}</strong>{link}<br>{preview}...</div>',
                        unsafe_allow_html=True
                    )

            n_sources = len(sources)
            if n_sources >= 3:
                st.markdown('<span class="confidence-high">● High confidence — well-supported by sources</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="confidence-low">● Moderate confidence — limited source coverage</span>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "blocked": was_blocked
    })
