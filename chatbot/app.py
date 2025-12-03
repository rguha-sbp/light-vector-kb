"""
Streamlit Chatbot UI (optional) for light-vector-kb

This UI wraps the existing knowledge base built on LanceDB and exposes
chat-style retrieval using strands-agents. It is intentionally optional:
if Streamlit or strands-agents are not available, the app will present
clear guidance instead of failing.
"""

import os
import sys
from pathlib import Path
import streamlit as st
from loguru import logger

try:
  from strands import Agent  # type: ignore
except Exception:
  Agent = None  # optional dependency; UI will degrade gracefully

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from config import C
from knowledge_base import create_knowledge_base


def load_css(css_path: str) -> None:
  try:
    css = Path(css_path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
  except Exception:
    logger.debug("CSS load failed or not found | path={}", css_path)


@st.cache_resource
def get_kb():
  logger.debug("Initializing knowledge base | table={} uri={} local_mode={}", C.LANCEDB_TABLE_NAME, C.LANCEDB_URI, C.USE_LOCAL_LANCEDB)
  kb = create_knowledge_base(
    documents_s3_bucket=C.DOCUMENTS_BUCKET_NAME,
    vector_s3_bucket=C.VECTOR_DATA_BUCKET_NAME,
    lance_table_name=C.LANCEDB_TABLE_NAME,
    chunking_strategy=C.CHUNKING_STRATEGY,
  )
  logger.info("Knowledge base ready | table={}", C.LANCEDB_TABLE_NAME)
  return kb


def retrieve_answer(kb, query: str, limit: int = 6) -> dict:
  """Use example.py-like hybrid search for knowledge retrieval."""
  try:
    logger.debug("KB search invoked | query='{}' limit={}", query, limit)
    result_json = kb.search_knowledge_base(query, limit=limit)
    logger.debug("KB search completed | length={}", len(result_json) if isinstance(result_json, str) else -1)
    return {"ok": True, "data": result_json}
  except Exception as e:
    logger.exception("KB search failed | query='{}' error={}", query, e)
    return {"ok": False, "error": str(e)}


def ensure_local_dirs():
  if C.USE_LOCAL_LANCEDB:
    Path(C.LOCAL_LANCEDB_PATH).resolve().mkdir(parents=True, exist_ok=True)
    Path(C.LOCAL_DOCUMENTS_PATH).resolve().mkdir(parents=True, exist_ok=True)
    logger.info("Ensured local directories | lancedb_path={} docs_path={}", C.LOCAL_LANCEDB_PATH, C.LOCAL_DOCUMENTS_PATH)


def render_header():
  logger.debug("Rendering header")
  st.markdown(
    """
    <div class="app-header">
      <h1>🗂️ Light Knowledge Base — Chatbot</h1>
      <p>Chat with your locally indexed documents (LanceDB + Bedrock).</p>
    </div>
    """,
    unsafe_allow_html=True,
  )


def render_sidebar():
  logger.debug("Rendering sidebar")
  with st.sidebar:
    st.header("Settings")
    st.write("Mode:")
    st.caption("Local mode is default. Configure via environment or config.")
    st.write({
      "USE_LOCAL_LANCEDB": C.USE_LOCAL_LANCEDB,
      "LANCEDB_URI": C.LANCEDB_URI,
      "LANCEDB_TABLE_NAME": C.LANCEDB_TABLE_NAME,
    })
    st.divider()
    st.subheader("Actions")
    if st.button("Process local documents"):
      kb = get_kb()
      ensure_local_dirs()
      logger.info("Starting document processing from sidebar")
      count = kb.process_documents(prefix="", publisher_system="chat-ui")
      st.success(f"Processed {count} chunks")
      logger.info("Document processing complete | chunks_processed={}", count)


def main():
  # Configure loguru sinks
  logger.remove()
  logger.add("chatbot.log", rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)
  logger.info("Chatbot app starting")

  st.set_page_config(page_title="Light Vector KB — Chatbot", layout="wide")
  load_css(str(Path(__file__).parent / "styles.css"))
  render_header()
  render_sidebar()

  if Agent is None:
    logger.warning("Strands Agent unavailable; presenting search-only UI")
    st.warning("Chat agent is optional. Install 'strands' to enable conversations.")
    st.info("You can still ingest and search via the sidebar.")
    kb = get_kb()
    query = st.text_input("Quick search", placeholder="Ask about your documents...")
    if query:
      res = retrieve_answer(kb, query)
      if res.get("ok"):
        st.json(res["data"])
      else:
        st.error(res.get("error"))
        logger.error("Quick search failed | error={}", res.get("error"))
    return

  # Build a strands-agents chat loop using AWS Bedrock (Amazon Nova Pro) by default
  llm_model = os.getenv("CHATBOT_LLM_MODEL", "eu.amazon.nova-pro-v1:0")
  aws_region = os.getenv("AWS_REGION", C.AWS_REGION)
  aws_profile = os.getenv("AWS_PROFILE") or os.getenv("AWS_DEFAULT_PROFILE")

  try:
    logger.info("Initializing Agent | model={} region={} profile={}", llm_model, aws_region, aws_profile or "default")
    agent = Agent(model=llm_model)
  except Exception as e:
    logger.exception("Agent init failed | model={} error={}", llm_model, e)
    st.error(f"Failed to initialize Agent with model '{llm_model}': {e}")
    st.stop()

  st.caption(f"Model: {llm_model} | Region: {aws_region} | Profile: {aws_profile or 'default'}")
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

  kb = get_kb()
  logger.debug("KB fetched for chat session")
  st.markdown("## Chat")
  user_input = st.chat_input("Type your question...")
  chat_container = st.container()

  # Display history
  with chat_container:
    for msg in st.session_state.chat_history:
      if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
      else:
        st.chat_message("assistant").write(msg["content"])

  if user_input:
    logger.debug("User input | text='{}'", user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Retrieve KB context JSON and fold into a single prompt per quickstart style
    retrieval = retrieve_answer(kb, user_input)
    kb_json = retrieval.get("data") if retrieval.get("ok") else None
    prompt = user_input
    if kb_json:
      prompt = (
        "You are a helpful assistant. Use the following knowledge base JSON to answer succinctly. "
        "Cite source file names when relevant.\n\nKB_CONTEXT_JSON:\n" + kb_json + "\n\nQuestion: " + user_input
      )

    try:
      result = agent(prompt)
      # strands Agent returns an AgentResult; extract assistant message text
      msg = getattr(result, "message", None)
      assistant_text = ""
      if isinstance(msg, dict):
        content = msg.get("content", [])
        # content is a list of segments; concatenate any 'text' fields
        texts = []
        for seg in content:
          if isinstance(seg, dict) and "text" in seg:
            texts.append(seg["text"])
        assistant_text = "\n".join(texts) if texts else str(msg)
      else:
        # Fallback to string representation
        assistant_text = str(result)
      logger.debug("Agent response length={}", len(assistant_text))
    except Exception as e:
      logger.exception("Agent run failed | error={}", e)
      assistant_text = f"Agent error: {e}\nKB Fallback: {kb_json or retrieval.get('error')}"

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
    st.chat_message("assistant").write(assistant_text)


if __name__ == "__main__":
  # Optional environment defaults for local runs
  os.environ.setdefault("AWS_REGION", C.AWS_REGION)
  main()
