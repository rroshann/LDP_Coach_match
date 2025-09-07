import io
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# NLP / ML
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Files
from pypdf import PdfReader
import docx  # python-docx

# Optimization
import pulp as pl

# Optional LLM (OpenAI)
try:
    import openai
except ImportError:
    openai = None


# ================== UI & PAGE ==================
st.set_page_config(page_title="LDP Matching v8 (Unsupervised + LLM)", layout="wide")
st.title("LDP Matching v8 (Unsupervised themes + optional LLM)")

# ================== HELPERS ==================
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def read_docx_text_filelike(file) -> str:
    doc = docx.Document(file)
    parts = [(p.text or "").strip() for p in doc.paragraphs]
    return "\n".join([p for p in parts if p])

def read_pdf_text_filelike(file) -> str:
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(text)
    except Exception:
        return ""

@st.cache_resource(show_spinner=False)
def get_model(name: str = "all-mpnet-base-v2"):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def cache_embedding(texts: Tuple[str, ...], model_name: str):
    model = get_model(model_name)
    vecs = model.encode(list(texts), normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    return np.array(vecs)

def tfidf_keyphrases_per_doc(texts: List[str], top_k: int = 15) -> List[List[str]]:
    """Extract top-k keyphrases per document via TF-IDF (1–3 grams)."""
    cleaned = [normalize_whitespace(t or "") for t in texts]
    vec = TfidfVectorizer(ngram_range=(1,3), stop_words="english", max_features=10000)
    X = vec.fit_transform(cleaned)
    feats = vec.get_feature_names_out()
    out = []
    for i in range(X.shape[0]):
        row = X[i].toarray()[0]
        idx = np.argsort(row)[::-1]
        phrases = [feats[j] for j in idx[:top_k] if row[j] > 0]
        phrases = [p.strip(" -_.").lower() for p in phrases if len(p) > 2]
        out.append(phrases)
    return out

def tfidf_top_terms_for_cluster(doc_indices: List[int], tfidf_matrix, feature_names, top_k: int = 12) -> List[str]:
    if not doc_indices:
        return []
    sub = tfidf_matrix[doc_indices, :]
    summed = np.asarray(sub.sum(axis=0)).ravel()
    idx = np.argsort(summed)[::-1][:top_k]
    terms = [feature_names[j] for j in idx if summed[j] > 0]
    return [t.strip(" -_.").lower() for t in terms]

def have_llm():
    return (openai is not None) and bool(st.session_state.get("OPENAI_API_KEY"))

def llm_label_cluster(model: str, terms: List[str]) -> str:
    if not have_llm():
        return ", ".join(terms[:3]) if terms else "general"
    openai.api_key = st.session_state.get("OPENAI_API_KEY")
    system = ("You are labeling themes for coaching/student documents. "
              "Given a list of key terms, produce a 3–5 word theme name. "
              "No punctuation except spaces; avoid sensitive or demographic references.")
    user = "Top terms:\n- " + "\n- ".join(terms[:12])
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0.2,
            max_tokens=16,
        )
        label = resp["choices"][0]["message"]["content"].strip()
        label = re.sub(r"[^A-Za-z0-9 \-/&]", "", label)
        return label[:40] if label else (", ".join(terms[:3]) or "general")
    except Exception:
        return ", ".join(terms[:3]) if terms else "general"

def llm_reason_from_terms(model: str, coach_alias: str, sid: str,
                          student_terms: List[str], coach_terms: List[str], theme_names: List[str]) -> str:
    if not have_llm():
        return f"Coach {coach_alias} and Student {sid} align on " + ", ".join(theme_names[:2]) + "."
    openai.api_key = st.session_state.get("OPENAI_API_KEY")
    system = (
        "Write ONE grounded, non-technical sentence (≤35 words) explaining why the coach fits the student. "
        "ONLY use the provided terms and theme names. Do NOT invent details or mention scores."
    )
    user = f"""
coach_alias: {coach_alias}
student_id: {sid}

theme_names:
- {chr(10).join(theme_names[:3]) if theme_names else "(none)"}

student_terms:
- {chr(10).join(student_terms[:6]) if student_terms else "(none)"}

coach_terms:
- {chr(10).join(coach_terms[:6]) if coach_terms else "(none)"}
"""
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0.25,
            max_tokens=64,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        text = re.sub(r"\s+", " ", text)
        if not text.endswith("."): text += "."
        return text
    except Exception:
        return f"Coach {coach_alias} and Student {sid} align on " + ", ".join(theme_names[:2]) + "."

def deterministic_template(sid: str, coach_alias: str, themes: List[str]) -> str:
    templates = [
        "Coach {coach} and Student {sid} align on {t1}{t2}.",
        "Student {sid} matches with Coach {coach} around {t1}{t2}.",
        "Coach {coach} suits Student {sid} given focus on {t1}{t2}.",
    ]
    key = f"{sid}|{coach_alias}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    tmpl = templates[h % len(templates)]
    t1 = themes[0] if themes else "shared priorities"
    t2 = f" and {themes[1]}" if len(themes) > 1 else ""
    return tmpl.format(coach=coach_alias, sid=sid, t1=t1, t2=t2)

# ================== THEME DISCOVERY (UNSUPERVISED) ==================
def discover_themes(all_texts: List[str], top_terms_k: int = 12,
                    k_min: int = 6, k_max: int = 18) -> Dict:
    """
    Cluster normalized embeddings with KMeans; choose k by silhouette.
    Returns dict with: labels, k, centroids, cluster_terms, cluster_names, tfidf artifacts.
    """
    vecs = cache_embedding(tuple(all_texts), model_name="all-mpnet-base-v2")
    # Normalize again (safety)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    X = vecs / norms

    best_k, best_score, best_model = None, -1, None
    n = X.shape[0]
    k_max = min(k_max, max(2, n-1))
    for k in range(max(2, k_min), k_max+1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:  # silhouette needs at least 2 clusters
                continue
            score = silhouette_score(X, labels, metric="euclidean")
            if score > best_score:
                best_k, best_score, best_model = k, score, km
        except Exception:
            continue

    if best_model is None:
        # Fallback small k
        best_k = 8 if n >= 8 else max(2, n//2)
        best_model = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)

    labels = best_model.labels_
    centroids = best_model.cluster_centers_  # already normalized-ish

    # TF-IDF over all docs, then get top terms per cluster
    cleaned = [normalize_whitespace(t or "") for t in all_texts]
    tfvec = TfidfVectorizer(ngram_range=(1,3), stop_words="english", max_features=10000)
    tfX = tfvec.fit_transform(cleaned)
    feats = tfvec.get_feature_names_out()

    cluster_terms = {}
    for cid in range(best_k):
        doc_ids = np.where(labels == cid)[0].tolist()
        terms = tfidf_top_terms_for_cluster(doc_ids, tfX, feats, top_k=top_terms_k)
        cluster_terms[cid] = terms

    return {
        "labels": labels,
        "k": best_k,
        "centroids": centroids,
        "tfidf_matrix": tfX,
        "feature_names": feats,
        "cluster_terms": cluster_terms,
    }

def theme_vectors_per_doc(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Doc x Themes matrix of cosine similarities to centroids."""
    # Normalize
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    cen = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    return np.clip(emb @ cen.T, -1.0, 1.0)

# ================== MATCHING (unchanged ILP) ==================
def ilp_assignment(score: np.ndarray,
                   student_ids: List[str],
                   coach_aliases: List[str],
                   min_per: Dict[str, int],
                   max_per: Dict[str, int],
                   forbidden_pairs: List[Tuple[int,int]] = None) -> Dict[str, str]:
    n_students, n_coaches = score.shape
    min_list = [min_per.get(a, 0) for a in coach_aliases]
    max_list = [max_per.get(a, 9999) for a in coach_aliases]
    if n_students < sum(min_list):
        min_list = [0]*len(coach_aliases)

    prob = pl.LpProblem("ldp_match", pl.LpMaximize)
    x = pl.LpVariable.dicts("x", (range(n_students), range(n_coaches)), 0, 1, cat="Binary")
    prob += pl.lpSum(score[i, j] * x[i][j] for i in range(n_students) for j in range(n_coaches))
    for i in range(n_students):
        prob += pl.lpSum(x[i][j] for j in range(n_coaches)) == 1
    for j in range(n_coaches):
        if min_list[j] > 0:
            prob += pl.lpSum(x[i][j] for i in range(n_students)) >= int(min_list[j])
        prob += pl.lpSum(x[i][j] for i in range(n_students)) <= int(max_list[j])

    forb = set(forbidden_pairs or [])
    for (i_forb, j_forb) in forb:
        prob += x[i_forb][j_forb] == 0

    status = prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[status] != "Optimal":
        used = {j: 0 for j in range(n_coaches)}
        assigned = {}
        pairs = [(i, j, score[i, j]) for i in range(n_students) for j in range(n_coaches) if (i, j) not in forb]
        pairs.sort(key=lambda t: t[2], reverse=True)
        for i, j, s in pairs:
            if i in assigned: continue
            if used[j] < max_list[j]:
                assigned[i] = j; used[j] += 1
        for i in range(n_students):
            if i not in assigned:
                for j in np.argsort(-score[i, :]):
                    if (i, j) in forb: continue
                    if used[j] < max_list[j]:
                        assigned[i] = j; used[j] += 1; break
        return {student_ids[i]: coach_aliases[assigned[i]] for i in range(n_students)}

    mapping = {}
    for i in range(n_students):
        for j in range(n_coaches):
            if pl.value(x[i][j]) > 0.5:
                mapping[student_ids[i]] = coach_aliases[j]; break
    return mapping

# ================== SIDEBAR: Uploads & Settings ==================
st.sidebar.header("Upload files")
survey_file = st.sidebar.file_uploader("Survey Excel (Sheet1 with 'Entry Id')", type=["xlsx", "xls"])
student_pdfs = st.sidebar.file_uploader("Student resumes (PDF, filenames EntryId.pdf)", type=["pdf"], accept_multiple_files=True)
coach_docs   = st.sidebar.file_uploader("Coach bios (DOCX)", type=["docx"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.header("Optional capacity CSVs")
cap_primary   = st.sidebar.file_uploader("Primary capacities (coach_alias,min,max)", type=["csv"])
cap_secondary = st.sidebar.file_uploader("Secondary capacities (coach_alias,min,max)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Matching weights")
w_text = st.sidebar.slider("Weight: Text similarity", 0.0, 1.0, 0.5, 0.05)
w_theme = 1.0 - w_text
st.sidebar.caption(f"Theme alignment weight auto-set to {w_theme:.2f}")

min_primary_default = st.sidebar.number_input("Primary min per coach", min_value=0, max_value=50, value=1)
max_primary_default = st.sidebar.number_input("Primary max per coach", min_value=1, max_value=50, value=7)
enable_secondary = st.sidebar.toggle("Enable Secondary Coach", value=True)
min_secondary_default = st.sidebar.number_input("Secondary min per coach", min_value=0, max_value=50, value=1)
max_secondary_default = st.sidebar.number_input("Secondary max per coach", min_value=1, max_value=50, value=7)

st.sidebar.markdown("---")
st.sidebar.header("LLM options")
use_llm_labels = st.sidebar.toggle("Use LLM for theme labels", value=False)
use_llm_reasons = st.sidebar.toggle("Use LLM for reasons", value=False)
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", help="Used only in this session; not saved.")
if api_key_input:
    st.session_state["OPENAI_API_KEY"] = api_key_input
llm_model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=not (use_llm_labels or use_llm_reasons))

# Theme discovery tuning
st.sidebar.markdown("---")
st.sidebar.header("Theme discovery (unsupervised)")
k_min = st.sidebar.number_input("Min clusters (k_min)", min_value=2, max_value=50, value=6)
k_max = st.sidebar.number_input("Max clusters (k_max)", min_value=3, max_value=60, value=18)

run_btn = st.sidebar.button("Run Matching", type="primary")

# ================== INGEST ==================
def ingest_coaches(files) -> pd.DataFrame:
    rows = []
    for f in files:
        alias = Path(f.name).stem
        text = normalize_whitespace(read_docx_text_filelike(f))
        rows.append({"coach_alias": alias, "coach_text": text})
    return pd.DataFrame(rows).sort_values("coach_alias").reset_index(drop=True)

def ingest_students_and_survey(student_files, survey_file) -> Tuple[pd.DataFrame, List[str], List[str]]:
    survey = pd.read_excel(survey_file, sheet_name="Sheet1")
    survey.columns = [c.strip() for c in survey.columns]
    if "Entry Id" not in survey.columns:
        raise ValueError("Survey must have a column named 'Entry Id'.")
    survey["Entry Id"] = survey["Entry Id"].astype(str).str.strip()

    pdf_text_by_id, pdf_ids = {}, []
    for f in student_files:
        sid = Path(f.name).stem
        pdf_ids.append(sid)
        pdf_text_by_id[sid] = normalize_whitespace(read_pdf_text_filelike(f))

    # Build student_text as resume + any free-text columns (we don't rely on fixed Likert mapping in v8)
    free_text_cols = [c for c in survey.columns if any(h in c.lower() for h in [
        "what do you want", "what other information", "career interests", "goals", "interests", "notes"
    ])]
    rows = []
    for _, row in survey.iterrows():
        sid = str(row["Entry Id"]).strip()
        resume_text = pdf_text_by_id.get(sid, "")
        free_text = " ".join(str(row.get(c, "")) for c in free_text_cols)
        student_text = normalize_whitespace("\n".join([resume_text, free_text]))
        rows.append({"Entry Id": sid, "student_text": student_text})
    students = pd.DataFrame(rows)

    survey_ids = survey["Entry Id"].astype(str).tolist()
    return students, survey_ids, pdf_ids

def load_caps(file, default_min: int, default_max: int, coach_aliases: List[str]):
    if file is None:
        return {a: default_min for a in coach_aliases}, {a: default_max for a in coach_aliases}
    caps = pd.read_csv(file)
    caps["coach_alias"] = caps["coach_alias"].astype(str)
    min_map = {a: default_min for a in coach_aliases}
    max_map = {a: default_max for a in coach_aliases}
    for _, r in caps.iterrows():
        a = r["coach_alias"]
        if a in min_map:
            if "min" in r and not pd.isna(r["min"]): min_map[a] = int(r["min"])
            if "max" in r and not pd.isna(r["max"]): max_map[a] = int(r["max"])
    return min_map, max_map

# ================== RUN ==================
results_df = None

if run_btn:
    if not (survey_file and student_pdfs and coach_docs):
        st.error("Please upload the survey Excel, student PDFs, and coach DOCX files.")
    else:
        with st.spinner("Ingesting files..."):
            coaches = ingest_coaches(coach_docs)
            students, survey_ids, pdf_ids = ingest_students_and_survey(student_pdfs, survey_file)
            st.session_state["coaches_df"] = coaches
            st.session_state["students_df"] = students

        # Sanity checks
        missing_pdfs = sorted(list(set(survey_ids) - set(pdf_ids)))
        extra_pdfs   = sorted(list(set(pdf_ids) - set(survey_ids)))
        with st.expander("Sanity checks", expanded=True):
            st.write(f"Coaches: **{len(coaches)}** | Students: **{len(students)}**")
            if missing_pdfs: st.warning(f"Missing PDF for: {missing_pdfs}")
            else: st.info("No missing PDFs detected.")
            if extra_pdfs: st.warning(f"PDF present but not in survey (ignored): {extra_pdfs}")
            else: st.info("No extra PDFs detected.")
            n_empty = int((students['student_text'].str.len() == 0).sum())
            if n_empty > 0: st.warning(f"{n_empty} student(s) have empty combined text (resume + free text).")

        # ===== Embeddings =====
        with st.spinner("Embedding texts..."):
            student_texts = students["student_text"].fillna("").tolist()
            coach_texts = coaches["coach_text"].fillna("").tolist()
            all_texts = student_texts + coach_texts
            all_vecs = cache_embedding(tuple(all_texts), model_name="all-mpnet-base-v2")
            S = all_vecs[:len(student_texts), :]
            C = all_vecs[len(student_texts):, :]

        # ===== Theme discovery (unsupervised) =====
        with st.spinner("Discovering themes (unsupervised clustering)..."):
            theme_obj = discover_themes(all_texts, top_terms_k=12, k_min=int(k_min), k_max=int(k_max))
            labels = theme_obj["labels"]
            k = theme_obj["k"]
            centroids = theme_obj["centroids"]
            cluster_terms = theme_obj["cluster_terms"]

            # Build TF-IDF artifacts also for reasons
            tfvec = TfidfVectorizer(ngram_range=(1,3), stop_words="english", max_features=10000)
            tfX = tfvec.fit_transform([normalize_whitespace(t) for t in all_texts])
            feats = tfvec.get_feature_names_out()

            # Cluster names (LLM if enabled)
            cluster_names = {}
            for cid in range(k):
                terms = cluster_terms.get(cid, [])
                label = llm_label_cluster(llm_model, terms) if use_llm_labels else (", ".join(terms[:3]) if terms else f"Theme {cid}")
                cluster_names[cid] = label

        # ===== Theme vectors per doc =====
        with st.spinner("Computing theme alignment vectors..."):
            theme_mat = theme_vectors_per_doc(all_vecs, centroids)  # (N_docs x k)
            stud_theme = theme_mat[:len(student_texts), :]
            coach_theme = theme_mat[len(student_texts):, :]

        # ===== Match scoring =====
        with st.spinner("Scoring pairs..."):
            sim_text = np.clip(cosine_similarity(S, C), -1.0, 1.0)
            sim_theme = np.clip(cosine_similarity(stud_theme, coach_theme), -1.0, 1.0)
            score = w_text * sim_text + w_theme * sim_theme

        # ===== Capacities =====
        coach_aliases = coaches["coach_alias"].tolist()
        min_primary_map, max_primary_map = load_caps(cap_primary, min_primary_default, max_primary_default, coach_aliases)
        min_secondary_map, max_secondary_map = load_caps(cap_secondary, min_secondary_default, max_secondary_default, coach_aliases)

        student_ids = students["Entry Id"].tolist()
        sid_to_i = {sid: i for i, sid in enumerate(student_ids)}
        alias_to_j = {alias: j for j, alias in enumerate(coach_aliases)}

        # ===== Primary assignment =====
        with st.spinner("Solving primary assignment..."):
            primary_map = ilp_assignment(score, student_ids, coach_aliases, min_primary_map, max_primary_map)

        # ===== Secondary assignment (optional) =====
        forbidden_pairs = [(sid_to_i[sid], alias_to_j[c]) for sid, c in primary_map.items()]
        if enable_secondary:
            with st.spinner("Solving secondary assignment..."):
                secondary_map = ilp_assignment(score, student_ids, coach_aliases, min_secondary_map, max_secondary_map, forbidden_pairs=forbidden_pairs)
        else:
            secondary_map = {}

        # ===== Reasons (LLM or template) =====
        # Build per-doc keyphrases for reasons (short, data-driven)
        doc_terms = tfidf_keyphrases_per_doc(all_texts, top_k=12)
        stud_terms = doc_terms[:len(student_texts)]
        coach_terms = doc_terms[len(student_texts):]

        rows = []
        for sid in student_ids:
            i = sid_to_i[sid]
            p_alias = primary_map[sid]
            pj = alias_to_j[p_alias]

            # Top theme names for this pair: intersect top student/coach theme indices by score
            s_theme_idx = np.argsort(-stud_theme[i])[:3].tolist()
            c_theme_idx = np.argsort(-coach_theme[pj])[:3].tolist()
            common_idx = [t for t in s_theme_idx if t in c_theme_idx]
            chosen_idx = (common_idx or s_theme_idx)[:2]
            theme_names = [cluster_names[t] for t in chosen_idx]

            if use_llm_reasons and have_llm():
                p_reason = llm_reason_from_terms(llm_model, p_alias, sid, stud_terms[i], coach_terms[pj], theme_names)
            else:
                p_reason = deterministic_template(sid, p_alias, theme_names)

            s_alias = secondary_map.get(sid, "")
            if s_alias:
                sj = alias_to_j[s_alias]
                s_theme_idx2 = np.argsort(-stud_theme[i])[:3].tolist()
                c_theme_idx2 = np.argsort(-coach_theme[sj])[:3].tolist()
                common_idx2 = [t for t in s_theme_idx2 if t in c_theme_idx2]
                chosen_idx2 = (common_idx2 or s_theme_idx2)[:2]
                theme_names2 = [cluster_names[t] for t in chosen_idx2]
                if use_llm_reasons and have_llm():
                    s_reason = llm_reason_from_terms(llm_model, s_alias, sid, stud_terms[i], coach_terms[sj], theme_names2)
                else:
                    s_reason = deterministic_template(sid, s_alias, theme_names2)
            else:
                s_reason = ""

            rows.append({
                "Entry Id": sid,
                "PrimaryCoach": p_alias,
                "PrimaryReason": p_reason,
                "SecondaryCoach": s_alias,
                "SecondaryReason": s_reason
            })

        results_df = pd.DataFrame(rows)

        # Persist for UI
        st.session_state["results_df"] = results_df
        st.session_state["coach_aliases"] = coach_aliases
        st.session_state["cluster_names"] = cluster_names
        st.session_state["stud_theme"] = stud_theme
        st.session_state["coach_theme"] = coach_theme
        st.session_state["stud_terms"] = stud_terms
        st.session_state["coach_terms"] = coach_terms
        st.session_state["llm_model"] = llm_model

# ================== REVIEW / RECOMPUTE / EXPORT ==================
results_df = st.session_state.get("results_df")
if results_df is not None:
    st.subheader("Review & Edit")
    st.caption("Edit coaches in-grid. Use **Recompute reasons** after edits.")

    coach_options = sorted(set(
        st.session_state.get("coach_aliases", [])
        + results_df["PrimaryCoach"].dropna().tolist()
        + results_df["SecondaryCoach"].fillna("").tolist()
    ))
    coach_options = [c for c in coach_options if c]

    edited = st.data_editor(
        results_df,
        key="results_editor",
        num_rows="fixed",
        column_config={
            "PrimaryCoach":   st.column_config.SelectboxColumn(options=coach_options),
            "SecondaryCoach": st.column_config.SelectboxColumn(options=[""] + coach_options),
        },
        use_container_width=True
    )
    st.session_state["results_df"] = edited

    def recompute_reasons():
        df = st.session_state.get("results_df")
        students = st.session_state.get("students_df")
        coaches = st.session_state.get("coaches_df")
        stud_theme = st.session_state.get("stud_theme")
        coach_theme = st.session_state.get("coach_theme")
        stud_terms = st.session_state.get("stud_terms")
        coach_terms = st.session_state.get("coach_terms")
        cluster_names = st.session_state.get("cluster_names", {})
        model_name = st.session_state.get("llm_model", "gpt-4o-mini")
        if df is None or students is None or coaches is None:
            st.warning("Run Matching first."); return

        sid_to_i = {sid: i for i, sid in enumerate(students["Entry Id"].tolist())}
        alias_to_j = {alias: j for j, alias in enumerate(coaches["coach_alias"].tolist())}

        new_rows = []
        for _, row in df.iterrows():
            sid = str(row["Entry Id"])
            i = sid_to_i.get(sid, None)
            p_alias = str(row.get("PrimaryCoach", "")) or ""
            s_alias = str(row.get("SecondaryCoach", "")) or ""

            # Primary
            if i is not None and p_alias in alias_to_j:
                pj = alias_to_j[p_alias]
                s_idx = np.argsort(-stud_theme[i])[:3].tolist()
                c_idx = np.argsort(-coach_theme[pj])[:3].tolist()
                common = [t for t in s_idx if t in c_idx]
                chosen = (common or s_idx)[:2]
                tnames = [cluster_names.get(t, f"Theme {t}") for t in chosen]
                if use_llm_reasons and have_llm():
                    p_reason = llm_reason_from_terms(model_name, p_alias, sid, stud_terms[i], coach_terms[pj], tnames)
                else:
                    p_reason = deterministic_template(sid, p_alias, tnames)
            else:
                p_reason = f"Coach {p_alias} aligns with Student {sid}."

            # Secondary
            if s_alias and i is not None and s_alias in alias_to_j:
                sj = alias_to_j[s_alias]
                s_idx2 = np.argsort(-stud_theme[i])[:3].tolist()
                c_idx2 = np.argsort(-coach_theme[sj])[:3].tolist()
                common2 = [t for t in s_idx2 if t in c_idx2]
                chosen2 = (common2 or s_idx2)[:2]
                tnames2 = [cluster_names.get(t, f"Theme {t}") for t in chosen2]
                if use_llm_reasons and have_llm():
                    s_reason = llm_reason_from_terms(model_name, s_alias, sid, stud_terms[i], coach_terms[sj], tnames2)
                else:
                    s_reason = deterministic_template(sid, s_alias, tnames2)
            else:
                s_reason = "" if not s_alias else f"Coach {s_alias} aligns with Student {sid}."

            new_rows.append({
                "Entry Id": sid,
                "PrimaryCoach": p_alias,
                "PrimaryReason": p_reason,
                "SecondaryCoach": s_alias,
                "SecondaryReason": s_reason
            })

        st.session_state["results_df"] = pd.DataFrame(new_rows)
        st.success("Reasons recomputed.")

    cols = st.columns(2)
    with cols[0]:
        st.button("Recompute reasons", on_click=recompute_reasons)
    with cols[1]:
        if (use_llm_labels or use_llm_reasons) and have_llm():
            st.info("LLM: ON (using sidebar key)")
        elif (use_llm_labels or use_llm_reasons):
            st.warning("LLM toggled ON but no key set.")
        else:
            st.caption("LLM: OFF (auto labels + template reasons)")

    st.subheader("Export")
    def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Matches") -> bytes:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
            df.to_excel(xw, sheet_name=sheet_name, index=False)
        return bio.getvalue()

    xlsx_bytes = to_excel_bytes(st.session_state["results_df"])
    st.download_button(
        "Download Excel (5 columns)",
        data=xlsx_bytes,
        file_name="ldp_matches_v8.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Upload files, set options, and click **Run Matching**.")
