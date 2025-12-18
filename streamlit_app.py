import io
import re
import hashlib
import json
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

# ================== CONSTANTS ==================
TEXT_TRUNCATION_LIMIT = 4000  # Max characters sent to LLM prompts
TOP_K_KEYPHRASES = 12         # Number of keyphrases per document
TOP_K_CLUSTER_TERMS = 12      # Number of terms per cluster
EMBEDDING_BATCH_SIZE = 16     # Batch size for sentence embeddings
TFIDF_MAX_FEATURES = 10000    # Max features for TF-IDF vectorizer
DEFAULT_K_MIN = 6             # Min clusters for theme discovery
DEFAULT_K_MAX = 18            # Max clusters for theme discovery

# OpenAI LLM - uses API key from session state (set by app.py login flow)
openai_client = None
openai_available = False

def init_openai():
    """Initialize OpenAI client from session state API key."""
    global openai_client, openai_available
    api_key = st.session_state.get("openai_api_key", "")
    if api_key:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=api_key)
            openai_available = True
        except ImportError:
            openai_client = None
            openai_available = False
    else:
        openai_client = None
        openai_available = False

# Note: init_openai() is called at the start of main matching logic, not here



# ================== UI & PAGE ==================
# Note: st.set_page_config is called in app.py, not here
st.title("LDP Coach Matcher")

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
    except Exception as e:
        # Store warning to show later (can't show st.warning during function call)
        filename = getattr(file, 'name', 'unknown')
        st.warning(f"⚠️ Could not read PDF '{filename}': {str(e)[:100]}")
        return ""

@st.cache_resource(show_spinner=False)
def get_model(name: str = "all-mpnet-base-v2"):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def cache_embedding(texts: Tuple[str, ...], model_name: str):
    model = get_model(model_name)
    vecs = model.encode(list(texts), normalize_embeddings=True, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)
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
    """Check if OpenAI is available."""
    return openai_available

def llm_label_cluster(model: str, terms: List[str]) -> str:
    """Generate a theme label using OpenAI."""
    if not have_llm() or not terms:
        return ", ".join(terms[:3]) if terms else "general"
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are labeling themes for coaching/student documents. Given a list of key terms, produce a 3-5 word theme name. No punctuation except spaces; avoid sensitive or demographic references."},
                {"role": "user", "content": "Top terms:\n- " + "\n- ".join(terms[:12])}
            ],
            temperature=0.2,
            max_tokens=20,
        )
        label = response.choices[0].message.content.strip()
        label = re.sub(r"[^A-Za-z0-9 \-/&]", "", label)
        return label[:40] if label else (", ".join(terms[:3]) or "general")
    except Exception as e:
        st.warning(f"OpenAI API error: {e}")
        return ", ".join(terms[:3]) if terms else "general"

def llm_reason_from_features(model: str, coach_alias: str, sid: str,
                              student_features: Dict, coach_features: Dict,
                              theme_names: List[str] = None) -> str:
    """Generate a match reason using LLM-extracted features."""
    if not have_llm():
        # Fallback template using features
        interests = student_features.get("career_interests", [])[:2]
        industries = coach_features.get("industries", [])[:2]
        if interests and industries:
            return f"Coach {coach_alias}'s expertise in {', '.join(industries)} aligns with Student {sid}'s interest in {', '.join(interests)}."
        return f"Coach {coach_alias} and Student {sid} share complementary professional backgrounds."
    
    # Build rich context from extracted features
    student_interests = ", ".join(student_features.get("career_interests", [])[:3]) or "not specified"
    student_current = ", ".join(student_features.get("current_industries", [])[:3]) or "not specified"
    student_goals = ", ".join(student_features.get("coaching_goals", [])[:3]) or "professional development"
    student_style = student_features.get("preferred_coaching_style", 5)
    student_style_desc = "supportive, gentle" if student_style <= 3 else ("direct, challenging" if student_style >= 7 else "balanced")
    
    coach_industries = ", ".join(coach_features.get("industries", [])[:4]) or "diverse industries"
    coach_strengths = ", ".join(coach_features.get("key_strengths", [])[:3]) or "executive coaching"
    coach_style = coach_features.get("coaching_style", 5)
    coach_style_desc = "supportive and patient" if coach_style <= 3 else ("direct and challenging" if coach_style >= 7 else "balanced")
    coach_populations = ", ".join(coach_features.get("target_populations", [])[:2]) or "professionals"
    
    prompt = f"""Write ONE specific, grounded sentence (30-40 words) explaining why this coach-student match is strong.
Use the ACTUAL details provided below. Be specific about industries, goals, and alignment.

STUDENT {sid}:
- Career interests: {student_interests}
- Current background: {student_current}
- Coaching goals: {student_goals}
- Prefers {student_style_desc} coaching style

COACH {coach_alias}:
- Industry expertise: {coach_industries}
- Key strengths: {coach_strengths}
- Coaching style: {coach_style_desc}
- Works well with: {coach_populations}

Write a single, specific sentence explaining why this coach fits this student:"""
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Write ONE specific sentence explaining why the coach fits the student. Reference actual industries, goals, and strengths. Be concrete and grounded."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"\s+", " ", text)
        if not text.endswith("."):
            text += "."
        return text
    except Exception as e:
        st.warning(f"OpenAI API error: {e}")
        interests = student_features.get("career_interests", [])[:2]
        industries = coach_features.get("industries", [])[:2]
        if interests and industries:
            return f"Coach {coach_alias}'s expertise in {', '.join(industries)} aligns with Student {sid}'s interest in {', '.join(interests)}."
        return f"Coach {coach_alias} and Student {sid} share complementary professional backgrounds."

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

# ================== LLM-POWERED FEATURE EXTRACTION ==================

def llm_extract_coach_features(coach_alias: str, coach_text: str) -> Dict:
    """
    Use LLM to extract structured features from coach bio.
    Returns dict of extracted features.
    """
    # Default/fallback features
    default_features = {
        "coaching_style": 5,
        "industries": [],
        "target_populations": ["any"],
        "experience_level_preference": "any",
        "dei_focus": False,
        "international_experience": False,
        "entrepreneur_experience": False,
        "key_strengths": []
    }
    
    if not openai_available or not coach_text.strip():
        return default_features
    
    prompt = f"""You are analyzing an executive coach bio to extract structured features for a matching system.

Given this coach bio, extract the following features as JSON:

1. coaching_style (integer 1-10): 
   - 1-3 = supportive, patient, nurturing, asks gentle questions
   - 4-6 = balanced, adaptable
   - 7-10 = direct, challenging, pushes hard, straightforward

2. industries (list of strings): Areas of expertise mentioned (e.g., "healthcare", "finance", "technology", "hr", "consulting")

3. target_populations (list of strings): Who they prefer coaching (e.g., "executives", "early career", "MBA students", "women", "minorities")

4. experience_level_preference (string): "any", "early", "mid", or "senior"

5. dei_focus (boolean): Do they specialize in diverse/underrepresented populations?

6. international_experience (boolean): Have they worked/coached internationally?

7. entrepreneur_experience (boolean): Experience with startups/founders?

8. key_strengths (list of strings): Top 3-5 coaching strengths mentioned

COACH BIO:
{coach_text[:TEXT_TRUNCATION_LIMIT]}

Respond with ONLY valid JSON, no explanation or markdown."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract structured features from coach bios. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        features = json.loads(content)
        
        # Ensure all expected keys exist with proper types
        result = {
            "coaching_style": int(features.get("coaching_style", 5)),
            "industries": list(features.get("industries", [])),
            "target_populations": list(features.get("target_populations", ["any"])),
            "experience_level_preference": str(features.get("experience_level_preference", "any")),
            "dei_focus": bool(features.get("dei_focus", False)),
            "international_experience": bool(features.get("international_experience", False)),
            "entrepreneur_experience": bool(features.get("entrepreneur_experience", False)),
            "key_strengths": list(features.get("key_strengths", []))
        }
        return result
        
    except Exception as e:
        st.warning(f"Coach {coach_alias} feature extraction error: {e}")
        return default_features


def llm_extract_student_features(sid: str, student_text: str) -> Dict:
    """
    Use LLM to extract structured features from student survey + resume.
    Returns dict of extracted features.
    """
    # Default/fallback features
    default_features = {
        "preferred_coaching_style": 5,
        "career_interests": [],
        "current_industries": [],
        "experience_level": "mid",
        "coaching_goals": [],
        "personality_traits": [],
        "international_background": False,
        "underrepresented_group": False
    }
    
    if not openai_available or not student_text.strip():
        return default_features
    
    prompt = f"""You are analyzing an MBA student's survey responses and resume to extract features for coach matching.

Given this student profile, extract the following features as JSON:

1. preferred_coaching_style (integer 1-10):
   - 1-3 = wants supportive, patient, gentle guidance
   - 4-6 = balanced, no strong preference
   - 7-10 = wants direct, challenging, pushes them hard

2. career_interests (list of strings): Target industries/roles they want (e.g., "investment banking", "consulting", "healthcare")

3. current_industries (list of strings): Industries from their work experience

4. experience_level (string): "early" (<3 yrs work), "mid" (3-8 yrs), or "senior" (8+ yrs)

5. coaching_goals (list of strings): What they want from coaching (e.g., "career transition", "leadership skills", "confidence")

6. personality_traits (list of strings): Key traits evident from their writing

7. international_background (boolean): Non-US origin or significant international experience?

8. underrepresented_group (boolean): Minority, female, immigrant, or veteran based on their profile?

STUDENT SURVEY + RESUME:
{student_text[:TEXT_TRUNCATION_LIMIT]}

Respond with ONLY valid JSON, no explanation or markdown."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract structured features from student profiles. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        features = json.loads(content)
        
        # Ensure all expected keys exist with proper types
        result = {
            "preferred_coaching_style": int(features.get("preferred_coaching_style", 5)),
            "career_interests": list(features.get("career_interests", [])),
            "current_industries": list(features.get("current_industries", [])),
            "experience_level": str(features.get("experience_level", "mid")),
            "coaching_goals": list(features.get("coaching_goals", [])),
            "personality_traits": list(features.get("personality_traits", [])),
            "international_background": bool(features.get("international_background", False)),
            "underrepresented_group": bool(features.get("underrepresented_group", False))
        }
        return result
        
    except Exception as e:
        st.warning(f"Student {sid} feature extraction error: {e}")
        return default_features


def compute_enhanced_score(
    text_sim: float,
    theme_sim: float,
    student_features: Dict,
    coach_features: Dict,
) -> float:
    """
    Compute enhanced match score using LLM-extracted features.
    
    Weights:
    - Style Compatibility: 30%
    - Industry Alignment: 25%
    - Semantic Similarity: 25%
    - Experience Fit: 10%
    - Special Match Bonus: 10%
    """
    
    # === Style Compatibility (30%) ===
    # Perfect match when both are same, worst when opposite ends
    student_style = student_features.get("preferred_coaching_style", 5)
    coach_style = coach_features.get("coaching_style", 5)
    style_score = 1.0 - abs(student_style - coach_style) / 10.0
    
    # === Industry Alignment (25%) ===
    # Check overlap between student interests and coach expertise
    student_interests = set(i.lower() for i in student_features.get("career_interests", []))
    student_current = set(i.lower() for i in student_features.get("current_industries", []))
    all_student_industries = student_interests | student_current
    coach_industries = set(i.lower() for i in coach_features.get("industries", []))
    
    if all_student_industries and coach_industries:
        # Jaccard-like similarity with partial matching
        matches = 0
        for si in all_student_industries:
            for ci in coach_industries:
                if si in ci or ci in si or si == ci:
                    matches += 1
                    break
        industry_score = min(1.0, matches / max(1, len(all_student_industries)))
    else:
        industry_score = 0.5  # Neutral if no data
    
    # === Semantic Similarity (25%) ===
    # Average of text and theme similarity
    semantic_score = (text_sim + theme_sim) / 2.0
    
    # === Experience Fit (10%) ===
    student_exp = student_features.get("experience_level", "mid")
    coach_pref = coach_features.get("experience_level_preference", "any")
    
    if coach_pref == "any":
        exp_score = 1.0
    elif coach_pref == "senior" and student_exp == "early":
        exp_score = 0.3  # Penalty for mismatch
    elif coach_pref == "senior" and student_exp == "mid":
        exp_score = 0.7
    elif coach_pref == student_exp:
        exp_score = 1.0
    else:
        exp_score = 0.8  # Slight penalty for other mismatches
    
    # === Special Match Bonus (10%) ===
    special_score = 0.5  # Default neutral
    
    # DEI match bonus
    if student_features.get("underrepresented_group", False) and coach_features.get("dei_focus", False):
        special_score += 0.25
    
    # International match bonus
    if student_features.get("international_background", False) and coach_features.get("international_experience", False):
        special_score += 0.25
    
    special_score = min(1.0, special_score)
    
    # === Weighted Composite ===
    score = (
        0.30 * style_score +
        0.25 * industry_score +
        0.25 * semantic_score +
        0.10 * exp_score +
        0.10 * special_score
    )
    
    return score


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

# Default configuration values
cap_primary = None
cap_secondary = None
w_text = 0.5
w_theme = 0.5
min_primary_default = 1
max_primary_default = 7
enable_secondary = True
min_secondary_default = 1
max_secondary_default = 7
use_llm_labels = False
use_llm_reasons = True
llm_model = "gpt-4o-mini"
k_min = DEFAULT_K_MIN
k_max = DEFAULT_K_MAX

# Debug mode toggle
st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=False, help="Show extracted AI features for verification")

run_btn = st.sidebar.button("Run Matching", type="primary")

# ================== INGEST ==================
def ingest_coaches(files: List) -> pd.DataFrame:
    """Ingest coach DOCX files and extract text."""
    rows = []
    for f in files:
        alias = Path(f.name).stem
        text = normalize_whitespace(read_docx_text_filelike(f))
        rows.append({"coach_alias": alias, "coach_text": text})
    return pd.DataFrame(rows).sort_values("coach_alias").reset_index(drop=True)

def ingest_students_and_survey(student_files, survey_file) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
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

    # Build student_text as resume + any free-text columns
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
    return students, survey_ids, pdf_ids, survey

def load_caps(file, default_min: int, default_max: int, coach_aliases: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Load coach capacity limits from CSV file."""
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
    # Initialize OpenAI client from session state API key
    init_openai()
    
    if not (survey_file and student_pdfs and coach_docs):
        st.error("Please upload the survey Excel, student PDFs, and coach DOCX files.")
    else:
        with st.spinner("Ingesting files..."):
            coaches = ingest_coaches(coach_docs)
            students, survey_ids, pdf_ids, survey_df = ingest_students_and_survey(student_pdfs, survey_file)
            st.session_state["coaches_df"] = coaches
            st.session_state["students_df"] = students

        # ===== LLM-Powered Feature Extraction =====
        st.write("🤖 **AI Feature Extraction** - Analyzing profiles...")
        
        # Extract coach features with progress bar
        coach_features_dict = {}
        coach_progress = st.progress(0, text="Analyzing coach profiles...")
        coach_aliases_list = coaches["coach_alias"].tolist()
        for idx, row_data in coaches.iterrows():
            alias = row_data["coach_alias"]
            coach_features_dict[alias] = llm_extract_coach_features(alias, row_data["coach_text"])
            coach_progress.progress((idx + 1) / len(coaches), text=f"Analyzing coach {idx + 1}/{len(coaches)}: {alias}")
        coach_progress.progress(1.0, text=f"✅ Extracted features for {len(coach_features_dict)} coaches")
        st.session_state["coach_features"] = coach_features_dict
        
        # Extract student features with progress bar
        student_features = {}
        student_progress = st.progress(0, text="Analyzing student profiles...")
        student_ids_list = students["Entry Id"].tolist()
        student_texts_list = students["student_text"].tolist()
        for idx, (sid, s_text) in enumerate(zip(student_ids_list, student_texts_list)):
            student_features[sid] = llm_extract_student_features(sid, s_text)
            student_progress.progress((idx + 1) / len(student_ids_list), text=f"Analyzing student {idx + 1}/{len(student_ids_list)}: {sid}")
        student_progress.progress(1.0, text=f"✅ Extracted features for {len(student_features)} students")
        st.session_state["student_features"] = student_features

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
            
            # Show feature extraction summary
            st.write(f"**AI-Powered Matching:** LLM extracted features for {len(student_features)} students and {len(coach_features_dict)} coaches.")

        # ===== DEBUG MODE: Show Extracted Features =====
        if debug_mode:
            st.subheader("🔧 Debug Mode: Extracted Features")
            
            # Coach features
            with st.expander(f"📋 Coach Features ({len(coach_features_dict)} coaches)", expanded=False):
                for alias, feats in sorted(coach_features_dict.items()):
                    st.markdown(f"**{alias}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"• Style: {feats.get('coaching_style', 'N/A')}/10")
                        st.write(f"• Industries: {', '.join(feats.get('industries', [])) or 'None'}")
                        st.write(f"• DEI Focus: {'Yes' if feats.get('dei_focus') else 'No'}")
                    with col2:
                        st.write(f"• Experience Pref: {feats.get('experience_level_preference', 'any')}")
                        st.write(f"• International: {'Yes' if feats.get('international_experience') else 'No'}")
                        st.write(f"• Strengths: {', '.join(feats.get('key_strengths', [])[:3]) or 'None'}")
                    st.markdown("---")
            
            # Student features
            with st.expander(f"📋 Student Features ({len(student_features)} students)", expanded=False):
                for sid, feats in sorted(student_features.items()):
                    st.markdown(f"**Student {sid}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"• Preferred Style: {feats.get('preferred_coaching_style', 'N/A')}/10")
                        st.write(f"• Career Interests: {', '.join(feats.get('career_interests', [])[:3]) or 'None'}")
                        st.write(f"• Experience: {feats.get('experience_level', 'mid')}")
                    with col2:
                        st.write(f"• Goals: {', '.join(feats.get('coaching_goals', [])[:2]) or 'None'}")
                        st.write(f"• International: {'Yes' if feats.get('international_background') else 'No'}")
                        st.write(f"• Underrepresented: {'Yes' if feats.get('underrepresented_group') else 'No'}")
                    st.markdown("---")

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
            theme_obj = discover_themes(all_texts, top_terms_k=TOP_K_CLUSTER_TERMS, k_min=int(k_min), k_max=int(k_max))
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

        # ===== Match scoring (ENHANCED) =====
        with st.spinner("Scoring pairs with enhanced matching..."):
            sim_text = np.clip(cosine_similarity(S, C), -1.0, 1.0)
            sim_theme = np.clip(cosine_similarity(stud_theme, coach_theme), -1.0, 1.0)
            
            # Build enhanced score matrix
            student_ids = students["Entry Id"].tolist()
            coach_aliases = coaches["coach_alias"].tolist()
            n_students = len(student_ids)
            n_coaches = len(coach_aliases)
            
            score = np.zeros((n_students, n_coaches))
            for i, sid in enumerate(student_ids):
                s_feat = student_features.get(sid, {})
                for j, alias in enumerate(coach_aliases):
                    c_feat = coach_features_dict.get(alias, {})
                    score[i, j] = compute_enhanced_score(
                        text_sim=sim_text[i, j],
                        theme_sim=sim_theme[i, j],
                        student_features=s_feat,
                        coach_features=c_feat,
                    )

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
        
        # Progress bar for AI generation
        if use_llm_reasons and have_llm():
            progress_text = "🤖 Generating AI-powered match reasons..."
            progress_bar = st.progress(0, text=progress_text)
        
        total_students = len(student_ids)
        for idx, sid in enumerate(student_ids):
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
                s_feat = student_features.get(sid, {})
                c_feat = coach_features_dict.get(p_alias, {})
                p_reason = llm_reason_from_features(llm_model, p_alias, sid, s_feat, c_feat, theme_names)
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
                    s_feat = student_features.get(sid, {})
                    c_feat = coach_features_dict.get(s_alias, {})
                    s_reason = llm_reason_from_features(llm_model, s_alias, sid, s_feat, c_feat, theme_names2)
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
            
            # Update progress bar
            if use_llm_reasons and have_llm():
                progress = (idx + 1) / total_students
                progress_bar.progress(progress, text=f"🤖 Generating reasons... ({idx + 1}/{total_students} students)")
        
        # Clear progress bar when done
        if use_llm_reasons and have_llm():
            progress_bar.progress(1.0, text="✅ AI reason generation complete!")

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
                    s_feat = student_features.get(sid, {})
                    c_feat = coach_features.get(p_alias, {})
                    p_reason = llm_reason_from_features(model_name, p_alias, sid, s_feat, c_feat, tnames)
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
                    s_feat = student_features.get(sid, {})
                    c_feat = coach_features.get(s_alias, {})
                    s_reason = llm_reason_from_features(model_name, s_alias, sid, s_feat, c_feat, tnames2)
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
        file_name="ldp_matches.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Upload files, set options, and click **Run Matching**.")
