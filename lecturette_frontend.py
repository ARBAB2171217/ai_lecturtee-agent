from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

from lecturette_backend import app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "lecturette"


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# MongoDB: Past Lecturettes
# -----------------------------
def list_past_lecturettes(limit: int = 50) -> List[dict]:
    """MongoDB se stored lecturettes fetch karo."""
    try:
        from pymongo import MongoClient

        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "lecturette_db")
        col_name = os.getenv("MONGODB_COLLECTION", "lecturettes")
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        col = client[db_name][col_name]
        docs = list(
            col.find(
                {},
                {"topic": 1, "lecturette": 1, "as_of": 1, "saved_at": 1, "_id": 1},
            )
            .sort("saved_at", -1)
            .limit(limit)
        )
        return docs
    except Exception as e:
        st.sidebar.caption(f"MongoDB fetch error: {e}")
        return []


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Lecturette Agent", layout="wide", page_icon="🎖️")

st.title("🎖️ Lecturette Writing Agent")
st.caption("AI-powered short instructional talks · Auto-saves to MongoDB · Semantic cache to save tokens")

with st.sidebar:
    st.header("📝 Generate Lecturette")
    topic = st.text_area(
        "Topic / Question",
        height=110,
        placeholder="e.g. Role of NBC Recce in nuclear contaminated area...",
    )
    as_of = st.date_input("As-of date", value=date.today())

    with st.expander("⚙️ Settings"):
        cache_threshold = st.slider(
            "Semantic Cache Threshold",
            min_value=0.80,
            max_value=0.99,
            value=float(os.getenv("CACHE_THRESHOLD", "0.92")),
            step=0.01,
            help="Similarity score above which cached lecturette is reused. Higher = stricter match.",
        )
        os.environ["CACHE_THRESHOLD"] = str(cache_threshold)

    run_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📚 Past Lecturettes")

    past_docs = list_past_lecturettes()
    if not past_docs:
        st.caption("No saved lecturettes found in MongoDB.")
        selected_doc = None
    else:
        options: List[str] = []
        doc_by_label: Dict[str, dict] = {}
        for doc in past_docs:
            md_text = doc.get("lecturette", "")
            title = extract_title_from_md(md_text, doc.get("topic", "Untitled"))
            saved_at = (doc.get("saved_at") or "")[:10]
            label = f"{title[:45]}  ·  {saved_at}"
            options.append(label)
            doc_by_label[label] = doc

        selected_label = st.radio(
            "Select to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_doc = doc_by_label.get(selected_label) if selected_label else None

        if st.button("📂 Load selected", use_container_width=True):
            if selected_doc:
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "final": selected_doc.get("lecturette", ""),
                    "cache_hit": False,
                    "mode": "loaded_from_db",
                }

# Session state init
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# Tabs
tab_preview, tab_plan, tab_evidence, tab_cache, tab_logs = st.tabs(
    ["📝 Lecturette", "🧩 Plan", "🔎 Evidence", "🧠 Cache Info", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


# -------------------------------------------------------
# Run Graph
# -------------------------------------------------------
if run_btn:
    if not topic.strip():
        st.warning("Pehle topic likhiye.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "topic_embedding": None,
        "cache_hit": False,
        "cached_lecturette": "",
        "augment_query": "",
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "final": "",
    }

    status = st.status("Running lecturette agent…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                node_labels = {
                    "semantic_cache": "🧠 Checking semantic cache…",
                    "augment": "⚡ Cache hit — augmenting…",
                    "save_augmented": "💾 Saving to MongoDB…",
                    "router": "🔀 Routing…",
                    "research": "🔎 Researching…",
                    "orchestrator": "🎯 Planning lecturette…",
                    "worker": "✍️ Writing sections…",
                    "assembler": "📦 Assembling + saving…",
                }
                status.write(node_labels.get(node_name, f"➡️ `{node_name}`"))
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "cache_hit": current_state.get("cache_hit"),
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "evidence_count": len(current_state.get("evidence", []) or []),
                "sections_done": len(current_state.get("sections", []) or []),
            }
            progress_area.json(summary)
            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            status.update(label="✅ Done", state="complete", expanded=False)
            log("[final] received")

# -------------------------------------------------------
# Render Output
# -------------------------------------------------------
out = st.session_state.get("last_out")
if out:
    final_md = out.get("final") or ""
    cache_hit = out.get("cache_hit", False)

    # --- Lecturette Preview ---
    with tab_preview:
        st.subheader("Lecturette")

        if cache_hit:
            st.success("⚡ **Cache Hit** — Retrieved from MongoDB (similar topic found). Only extra info was fetched.")

        if not final_md:
            st.warning("No lecturette generated.")
        else:
            st.markdown(final_md)

            blog_title = extract_title_from_md(final_md, "lecturette")
            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

    # --- Plan ---
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan available (cache hit or loaded from DB).")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("lecturette_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Kind:** " + str(plan_dict.get("lecturette_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence (closed_book mode, cache hit, or no Tavily key).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Cache Info ---
    with tab_cache:
        st.subheader("🧠 Semantic Cache Info")
        st.markdown(
            """
**How it works:**
1. Jab aap topic enter karte ho, uska **embedding** (768-dim vector) generate hota hai via Google `text-embedding-004`
2. MongoDB mein stored sabhi lecturettes ke embeddings se **cosine similarity** check hoti hai
3. Agar similarity ≥ threshold → **Cache Hit**: sirf thoda extra Tavily search + LLM augmentation
4. Agar nahi milta → **Full Pipeline**: Router → Research → Plan → Workers → Assemble → Save

**Benefits:**
- 🪙 LLM tokens bachte hain (full pipeline skip)
- ⚡ Response time kam hota hai
- 🔁 Duplicate content nahi banta MongoDB mein

**Threshold:** `{:.2f}` (sidebar se adjust kar sakte ho)
""".format(
                float(os.getenv("CACHE_THRESHOLD", "0.92"))
            )
        )

        # Show MongoDB stats
        try:
            from pymongo import MongoClient

            uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            db_name = os.getenv("MONGODB_DB", "lecturette_db")
            col_name = os.getenv("MONGODB_COLLECTION", "lecturettes")
            client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            col = client[db_name][col_name]
            total = col.count_documents({})
            with_embeddings = col.count_documents({"embedding": {"$exists": True, "$ne": []}})
            st.metric("Total Lecturettes in MongoDB", total)
            st.metric("With Embeddings (cache-eligible)", with_embeddings)
        except Exception as e:
            st.warning(f"MongoDB stats unavailable: {e}")

    # --- Logs ---
    with tab_logs:
        st.subheader("Logs")
        if logs:
            st.session_state["logs"].extend(logs)
        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=500)

else:
    with tab_preview:
        st.info("Topic enter karke **Generate** dabao.")
