from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Lecturette Writer Agent
# Flow: SemanticCache → (Router → Research? → Orchestrator → Workers → Assembler)
#       → MongoDB Store
#
# Semantic Cache:
#   - Har naye topic ka embedding generate hota hai
#   - MongoDB mein stored embeddings se cosine similarity check hoti hai
#   - Agar similarity >= CACHE_THRESHOLD: cached lecturette return + sirf
#     topic-specific extra info Tavily se fetch karke augment karo
#   - Warna: poora pipeline chalao
# ============================================================

# -----------------------------
# MongoDB + Embedding Setup
# -----------------------------
import math


def _get_mongo_collection():
    """Returns the MongoDB collection for lecturettes."""
    try:
        from pymongo import MongoClient

        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "lecturette_db")
        col_name = os.getenv("MONGODB_COLLECTION", "lecturettes")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        return client[db_name][col_name]
    except Exception as e:
        print(f"[MongoDB] Connection failed: {e}")
        raise RuntimeError("Embedding generation failed. Check GOOGLE_API_KEY or model availability.")


def _get_embedding(text: str) -> Optional[List[float]]:
    """
    Google Generative AI embedding model se text ka embedding vector banata hai.
    Falls back to None if unavailable.
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="SEMANTIC_SIMILARITY",
        )
        return result["embedding"]
    except Exception as e:
        print(f"[Embedding] Failed: {e}")
        raise RuntimeError("Embedding generation failed. Check GOOGLE_API_KEY or model availability.")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Similarity threshold: 0.92+ means almost same topic
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.92"))


def find_cached_lecturette(topic: str, embedding: List[float]) -> Optional[dict]:
    """
    MongoDB mein stored embeddings ke saath cosine similarity check karta hai.
    Returns the best matching document if similarity >= CACHE_THRESHOLD.
    """
    col = _get_mongo_collection()
    if col is None:
        raise RuntimeError("Embedding generation failed. Check GOOGLE_API_KEY or model availability.")
    try:
        # Exact match fast path
        exact_doc = col.find_one({"topic": topic})
        if exact_doc:
            print("[Cache HIT] Exact topic match found")
            return exact_doc

        docs = list(col.find({}, {"topic": 1, "embedding": 1, "lecturette": 1, "as_of": 1}))
        best_doc = None
        best_score = 0.0
        for doc in docs:
            stored_emb = doc.get("embedding")
            if not stored_emb:
                continue
            score = _cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_doc = doc
        if best_score >= CACHE_THRESHOLD and best_doc:
            print(f"[Cache HIT] similarity={best_score:.4f} for topic='{best_doc.get('topic')}'")
            return best_doc
        print(f"[Cache MISS] best_score={best_score:.4f}")
        raise RuntimeError("Embedding generation failed. Check GOOGLE_API_KEY or model availability.")
    except Exception as e:
        print(f"[Cache] Lookup failed: {e}")
        raise RuntimeError("Embedding generation failed. Check GOOGLE_API_KEY or model availability.")


def save_lecturette_to_mongo(
    topic: str,
    lecturette_md: str,
    embedding: List[float],
    plan: Optional[dict] = None,
    evidence: Optional[list] = None,
    as_of: str = "",
):
    """
    Lecturette ko MongoDB mein store karta hai — topic, embedding, full markdown.
    """
    col = _get_mongo_collection()
    if col is None:
        print("[MongoDB] Skipping save — no connection.")
        return
    try:
        from datetime import datetime

        doc = {
            "topic": topic,
            "topic_hash": topic_hash(topic),
            "lecturette": lecturette_md,
            "embedding": embedding,
            "as_of": as_of,
            "saved_at": datetime.utcnow().isoformat(),
            "plan": plan,
            "evidence_count": len(evidence) if evidence else 0,
        }
        col.update_one(
        {"topic_hash": topic_hash(topic)},
        {"$set": doc},
        upsert=True
    )
        print(f"[MongoDB] Saved lecturette with id={result.inserted_id}")
    except Exception as e:
        print(f"[MongoDB] Save failed: {e}")



# -----------------------------
# Topic Hash (duplicate prevention)
# -----------------------------
def topic_hash(topic: str) -> str:
    import hashlib
    return hashlib.sha256(
        topic.lower().strip().encode()
    ).hexdigest()

# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the learner should understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (150–400).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False


class Plan(BaseModel):
    lecturette_title: str
    audience: str
    tone: str
    lecturette_kind: Literal["concept_explainer", "procedure", "case_study", "comparison", "news_brief"] = "concept_explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # semantic cache
    topic_embedding: Optional[List[float]]
    cache_hit: bool
    cached_lecturette: str
    augment_query: str  # extra search query for cache-hit case

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]

    # final
    merged_md: str
    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key="AIzaSyC4CMqrN1xqioBRLvDcUOaZOkmDKXWLyZQ"
)

# -----------------------------
# 3) Semantic Cache Node
# -----------------------------
def semantic_cache_node(state: State) -> dict:
    """
    Topic ka embedding banao.
    MongoDB mein similarity check karo.
    Cache hit milne par: cached lecturette + augment_query set karo.
    """
    topic = state["topic"]
    embedding = _get_embedding(topic)

    if embedding is None:
        # Embedding unavailable → skip cache
        return {
            "topic_embedding": [],
            "cache_hit": False,
            "cached_lecturette": "",
            "augment_query": "",
        }

    cached = find_cached_lecturette(topic, embedding)
    if cached:
        # Cache hit: augment query = original topic (to fetch only what's extra/new)
        return {
            "topic_embedding": embedding,
            "cache_hit": True,
            "cached_lecturette": cached.get("lecturette", ""),
            "augment_query": topic,
        }

    return {
        "topic_embedding": embedding,
        "cache_hit": False,
        "cached_lecturette": "",
        "augment_query": "",
    }


def route_after_cache(state: State) -> str:
    return "augment" if state["cache_hit"] else "router"


# -----------------------------
# 4) Augment Node (Cache Hit Path)
# -----------------------------
AUGMENT_SYSTEM = """
You are a professional knowledge augmentation assistant.

A cached lecturette already exists for a similar topic.

Your job is to intelligently update or extend the lecturette for the NEW topic.

Rules:

1. Compare the NEW topic with the cached lecturette.
2. Identify missing, updated, or additional information.
3. If the cached lecturette already fully answers the topic:
   - Return the lecturette as-is.
   - Only update the title if needed.
4. If additional information is required:
   - Add a concise section titled:

## Additional Updates

This section must:

- Include new facts, statistics, or developments
- Reflect the latest situation or changes
- Maintain the same structure and tone
- Be professional and presentation-ready
- Be limited to about 150–200 words

Topic-specific requirements:

If topic involves:

Countries:
Include new developments in relations or policy changes.

Crime:
Include latest statistics or trends.

Economics:
Include recent data, growth rates, or indicators.

Technology:
Include new innovations or risks.

Do NOT rewrite the entire lecturette.

Only update or append missing information.

Output:

Full lecturette in Markdown format.
You are a sharp military/professional instructor assistant.

A cached lecturette exists for a similar topic. Your job:
1. Look at the NEW topic query vs the cached lecturette.
2. Identify what is DIFFERENT or EXTRA in the new query that is NOT covered by the cached lecturette.
3. If the cached lecturette fully answers the new topic: return it as-is with only the title updated.
4. If extra info is needed: append a short "## Additional Notes" section (max 150 words) with the new information.
   Use the provided web evidence snippets for this extra section.
5. Do NOT rewrite the entire lecturette. Only augment.

Output: full lecturette markdown (modified or as-is).
"""

def augment_node(state: State) -> dict:
    """
    Cache hit case mein: quick Tavily search + LLM se sirf extra part generate karo.
    """
    topic = state["topic"]
    cached = state["cached_lecturette"]
    augment_query = state.get("augment_query", topic)

    # Quick web fetch for any extra/new info
    raw_snippets = []
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tool = TavilySearchResults(max_results=3)
            results = tool.invoke({"query": augment_query}) or []
            for r in results:
                snippet = r.get("content") or r.get("snippet") or ""
                if snippet:
                    raw_snippets.append(snippet[:400])
        except Exception:
            pass

    snippets_text = "\n---\n".join(raw_snippets) if raw_snippets else "No additional web evidence found."

    augmented = llm.invoke(
        [
            SystemMessage(content=AUGMENT_SYSTEM),
            HumanMessage(
                content=(
                    f"New topic query: {topic}\n\n"
                    f"Cached lecturette:\n{cached}\n\n"
                    f"Extra web snippets:\n{snippets_text}"
                )
            ),
        ]
    ).content.strip()

    return {"final": augmented}


# -----------------------------
# 5) Router
# -----------------------------
ROUTER_SYSTEM = """
You are an intelligent routing module for a universal lecturette generation system.

Your task is to decide whether external research is required before generating a lecturette.

Evaluate the topic and determine:

- Does the topic require recent data?
- Does it involve statistics or current events?
- Does it involve policies, countries, crime, economics, or technology?

Modes:

closed_book
Use internal knowledge only.

hybrid
Use internal knowledge plus some current data.

open_book
Use extensive research for recent developments or statistics.

If research is required:

Generate 3 to 6 focused search queries.

Queries must retrieve:

- recent statistics
- official reports
- historical background
- current developments
- reliable data sources

Always prioritize:

Accuracy
Recency
Credibility
You are a routing module for a lecturette planner.

Decide whether web research is needed BEFORE planning a lecturette.

A lecturette is a short, focused instructional talk (5-10 min) used in military/professional training.

Modes:
- closed_book (needs_research=false): evergreen concepts, doctrine, procedures.
- hybrid (needs_research=true): concepts that need current examples/tools/recent events.
- open_book (needs_research=true): recent news, current operations, policy updates.

If needs_research=true: output 3–6 focused search queries.
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    recency_days = 7 if decision.mode == "open_book" else (45 if decision.mode == "hybrid" else 3650)

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 6) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        raise RuntimeError("Embedding generation failed. Check GOOGLE_API_KEY or model availability.")
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        raise RuntimeError("Embedding generation failed. Check GOOGLE_API_KEY or model availability.")


RESEARCH_SYSTEM = """
You are a research intelligence system.

Your job is to extract reliable factual evidence from web search results.

Always prioritize:

Government reports
Official statistics
Research institutions
International organizations
Credible news agencies

Rules:

Include only reliable sources.

Prefer:

UN
World Bank
Government
Academic journals
Major news agencies

Normalize dates.

Keep evidence concise.

Remove duplicates.
You are a research synthesizer for a lecturette planner.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer authoritative/official sources.
- Normalize published_at to ISO YYYY-MM-DD if inferable; else null.
- Keep snippets short and relevant.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:8]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=5))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}


# -----------------------------
# 7) Orchestrator
# -----------------------------
ORCH_SYSTEM = """
You are a senior knowledge architect and lecturette designer.

Your job is to create a structured outline for a professional lecturette on any topic.

Your lecturette structure must include:

1 Introduction
2 Background or history
3 Current situation or analysis
4 Data and statistics
5 Implications or challenges
6 Conclusion and key takeaways

Rules:

Create 4 to 7 sections.

Each section must contain:

- clear goal
- 3 to 6 bullet points
- logical progression

Mandatory topic-specific rules:

If topic involves:

Countries:
Include historical relations, current scenario, and strategic outlook.

Crime:
Include at least 5 years of statistical data.

Economics:
Include trends, comparisons, and growth rates.

Technology:
Include applications, benefits, risks, and future potential.

Social issues:
Include causes, impact, and policy responses.

Output must strictly follow the Plan schema.
You are a senior military/professional training curriculum designer.

Produce a highly structured outline for a LECTURETTE (short 5-10 minute instructional talk).

A lecturette is:
- Focused, tight, instructional — not a blog or essay
- Used in military/professional training settings
- Has clear learning objectives per section
- Ends with a summary or key takeaway

Requirements:
- 3–6 tasks (sections), each with goal + 3–5 bullets + target_words (150–400 per section)
- lecturette_kind: concept_explainer | procedure | case_study | comparison | news_brief
- Tone: formal/instructional for military audiences

Output must match Plan schema exactly.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:12]}"
                )
            ),
        ]
    )

    return {"plan": plan}


# -----------------------------
# 8) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# -----------------------------
# 9) Worker
# -----------------------------
WORKER_SYSTEM = """
You are a professional lecturette writer and subject matter expert.

Write ONE section of a lecturette.

Your writing must be:

Clear
Structured
Fact-based
Professional
Presentation-ready

Each section must:

Start with the key idea
Explain the concept clearly
Provide examples or data where relevant
Maintain logical flow
End with a meaningful insight

Mandatory rules:

Cover ALL bullets provided.

If the topic involves:

Crime:
Provide statistics for at least the last 5 years.

Countries:
Include historical context and current relations.

Economics:
Provide data trends and comparisons.

Technology:
Explain real-world applications.

Policy:
Explain impact and challenges.

Use numbers, percentages, and facts whenever possible.

Do not write vague statements.

Avoid fluff.

Output format:

Markdown

Start with:

## Section Title
You are an experienced military/professional training instructor.
Write ONE section of a lecturette in Markdown.

A lecturette section must be:
- Instructional, clear, direct — no fluff
- Written for an educated but non-specialist military/professional audience
- Structured: start with the key point, explain, give example if applicable
- End the last section (if it's the summary) with 1–2 actionable takeaways

Constraints:
- Cover ALL bullets in order
- Target words ±15%
- Output only section markdown starting with "## <Section Title>"
- No images, no decorative elements
- If requires_citations=True: cite Evidence URLs as Markdown links for factual claims
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:15]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Lecturette title: {plan.lecturette_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Lecturette kind: {plan.lecturette_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# -----------------------------
# 10) Assembler (merge + save)
# -----------------------------
def assembler_node(state: State) -> dict:
    """
    Sections ko order se merge karo.
    Final lecturette MongoDB mein save karo.
    Embedding use karo jo pehle se calculate ho chuki hai.
    """
    plan = state["plan"]
    if plan is None:
        raise ValueError("assembler_node called without plan.")

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.lecturette_title}\n\n{body}\n"

    # MongoDB mein save karo
    embedding = state.get("topic_embedding") or []
    if not embedding:
        embedding = _get_embedding(state["topic"])
    save_lecturette_to_mongo(
        topic=state["topic"],
        lecturette_md=final_md,
        embedding=embedding if embedding else (_get_embedding(state["topic"]) or []),
        plan=plan.model_dump() if plan else None,
        evidence=state.get("evidence", []),
        as_of=state.get("as_of", ""),
    )

    return {"merged_md": final_md, "final": final_md}


# ============================================================
# 11) MongoDB Save Node for augment path
# ============================================================
def save_augmented_node(state: State) -> dict:
    """
    Cache hit + augment path ke baad bhi MongoDB mein updated version save karo
    (sirf agar augmentation ne kuch add kiya ho).
    """
    final_md = state.get("final", "")
    cached_md = state.get("cached_lecturette", "")

    # Only save if something new was added
    if final_md and final_md.strip() != cached_md.strip():
        embedding = state.get("topic_embedding") or []
    if not embedding:
        embedding = _get_embedding(state["topic"])
        save_lecturette_to_mongo(
            topic=state["topic"],
            lecturette_md=final_md,
            embedding=embedding if embedding else (_get_embedding(state["topic"]) or []),
            as_of=state.get("as_of", ""),
        )

    return {}  # final already set


# ============================================================
# 12) Build Graph
# ============================================================
g = StateGraph(State)

# Nodes
g.add_node("semantic_cache", semantic_cache_node)
g.add_node("augment", augment_node)
g.add_node("save_augmented", save_augmented_node)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("assembler", assembler_node)

# Edges
g.add_edge(START, "semantic_cache")
g.add_conditional_edges(
    "semantic_cache",
    route_after_cache,
    {"augment": "augment", "router": "router"},
)

# Cache hit path
g.add_edge("augment", "save_augmented")
g.add_edge("save_augmented", END)

# Full pipeline path
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "assembler")
g.add_edge("assembler", END)

app = g.compile()



# -----------------------------
# Topic Type Detection (Part 9)
# -----------------------------
def detect_topic_type(topic: str) -> str:

    t = topic.lower()

    if "crime" in t or "women" in t or "violence" in t:
        return "crime"

    if "relation" in t or "country" in t or "india" in t:
        return "international_relations"

    if "economy" in t or "gdp" in t or "inflation" in t:
        return "economics"

    if "technology" in t or "ai" in t or "cyber" in t:
        return "technology"

    return "general"


# ============================================================
# Advanced Enhancements: Chunking + Vector Index + Hybrid Retrieval
# ============================================================

def chunk_lecturette_sections(text: str):
    """Split lecturette into section-level chunks using Markdown headers."""
    chunks = []
    sections = text.split("## ")
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        title = sec.split("\n")[0]
        chunks.append({
            "section_title": title,
            "content": sec
        })
    return chunks


def ensure_vector_index():
    """Create vector index if not exists (MongoDB)."""
    try:
        col = _get_mongo_collection()
        if col is None:
            return
        indexes = col.index_information()
        if "embedding_vector_index" not in indexes:
            col.create_index(
                [("embedding", "cosine")],
                name="embedding_vector_index"
            )
            print("[MongoDB] Vector index created.")
    except Exception as e:
        print(f"[MongoDB] Vector index creation failed: {e}")


def hybrid_retrieve(topic: str, embedding):
    """Hybrid retrieval: exact match first, then semantic similarity."""
    col = _get_mongo_collection()
    if col is None:
        return None

    # Exact match
    exact_doc = col.find_one({"topic": topic})
    if exact_doc:
        print("[Hybrid Retrieval] Exact match hit.")
        return exact_doc

    # Semantic search
    try:
        docs = list(col.find({}, {"topic": 1, "embedding": 1, "lecturette": 1}))
        best_doc = None
        best_score = 0.0
        for doc in docs:
            stored_emb = doc.get("embedding")
            if not stored_emb:
                continue
            score = _cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc and best_score >= CACHE_THRESHOLD:
            print(f"[Hybrid Retrieval] Semantic hit: {best_score:.4f}")
            return best_doc

    except Exception as e:
        print(f"[Hybrid Retrieval] Failed: {e}")

    return None


def token_saving_strategy(state: dict):
    """Skip heavy pipeline if sufficient cached content exists."""
    if state.get("cache_hit"):
        print("[Token Saver] Using cached lecturette. Skipping research.")
        return True
    return False

