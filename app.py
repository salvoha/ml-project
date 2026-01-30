from pathlib import Path
import difflib

import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Kickstarter Predictor", layout="centered")
st.title("Kickstarter Funding Prediction")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "model_lr1_final_no_backers.joblib"
PRE_PATH = BASE / "models" / "preprocessor3_adv.joblib"  # fallback only

# Hidden feature values
BACKERS_HIDDEN_DEFAULT = 0
PLEDGED_HIDDEN_DEFAULT = 0.0  # if your model expects it
GOAL_FULFILLMENT_DEFAULT = 0.0  # if your model expects it

# UI dropdowns
CATEGORIES = [
    "Fashion", "Film & Video", "Art", "Technology", "Journalism",
    "Publishing", "Theater", "Music", "Photography", "Games", "Design",
    "Food", "Crafts", "Comics", "Dance"
]

# Keep raw training strings (you had a trailing space on Netherlands)
COUNTRIES_RAW = [
    "United States", "United Kingdom", "Canada", "Australia",
    "New Zealand", "Netherlands ", "Sweden", "Denmark", "Norway",
    "Ireland", "Germany", "France", "Spain", "Belgium", "Italy",
    "Switzerland", "Austria", "Luxembourg", "Singapore", "Hong Kong",
    "Mexico", "Japan"
]
COUNTRIES_DISPLAY = [c.strip() for c in COUNTRIES_RAW]
DISPLAY_TO_RAW_COUNTRY = {c.strip(): c for c in COUNTRIES_RAW}

# Subcategory list (optional file override)
SUBCATS_FILE = BASE / "models" / "subcategories.txt"
SUBCATEGORIES_FALLBACK = [
    "Children's Books", "Crafts", "Jazz", "Music", "Comics",
    "Narrative Film", "Tabletop Games", "Digital Art", "Animation",
    "Conceptual Art", "Pop", "Hip-Hop", "Country & Folk",
    "Periodicals", "Webseries", "Performance Art", "Technology",
    "Art Books", "World Music", "Knitting", "Classical Music",
    "Poetry", "Graphic Novels", "Radio & Podcasts", "Design",
    "Hardware", "Webcomics", "Dance", "Translations", "Crochet",
    "Games", "Photo", "Mixed Media", "Space Exploration", "Photobooks",
    "Musical", "Audio", "Community Gardens", "R&B",
    "Fabrication Tools", "Textiles", "Architecture", "Immersive",
    "Literary Journals", "Spaces", "Video", "Apps", "DIY Electronics",
    "Academic", "Experimental", "Anthologies", "Plays", "Video Art",
    "Comic Books", "Letterpress", "Couture", "Robots", "Festivals",
    "Installations", "Sound", "Typography", "Stationery",
    "Camera Equipment", "Horror", "Flight", "Residencies", "Workshops",
    "Chiptune", "Civic Design", "Weaving", "Young Adult", "Web",
    "Makerspaces", "Glass", "Quilts", "Pottery", "Romance", "Ceramics",
    "Embroidery", "Candles", "DIY", "Printing", "Gadgets", "Zines",
    "Kids", "Footwear", "Pet Fashion", "Events", "Thrillers",
    "Woodworking", "Animals", "Vegan", "Ready-to-wear",
    "Gaming Hardware", "Movie Theaters", "Accessories", "Punk",
    "Metal", "Bacon", "Family", "Food Trucks", "Fantasy", "Places",
    "Live Games", "Science Fiction", "Drama", "Latin", "Calendars",
    "Television", "Music Videos", "Apparel", "Comedy", "Faith",
    "Restaurants", "Nature", "Cookbooks", "Drinks", "3D Printing",
    "People", "Childrenswear", "Farmer's Markets", "Jewelry",
    "Interactive Design", "Fine Art", "Blues", "Wearables",
    "Small Batch", "Farms", "Action", "Performances", "Playing Cards",
    "Mobile Games", "Taxidermy", "Print", "Literary Spaces"
]


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # preprocessor is optional fallback (only used if model is NOT a pipeline)
    pre = None
    if PRE_PATH.exists():
        pre = joblib.load(PRE_PATH)

    return model, pre


model, preprocessor = load_artifacts()


# -----------------------------
# Helpers
# -----------------------------
def load_subcategories():
    if SUBCATS_FILE.exists():
        lines = [ln.strip() for ln in SUBCATS_FILE.read_text(encoding="utf-8").splitlines()]
        lines = [x for x in lines if x]
        seen, out = set(), []
        for x in lines:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    return SUBCATEGORIES_FALLBACK


SUBCATEGORIES = load_subcategories()


def suggest_subcats(query: str, options: list[str], limit: int = 10) -> list[str]:
    q = (query or "").strip().lower()
    if not q:
        return []
    # substring matches first
    sub = [o for o in options if q in o.lower()]
    sub = sorted(sub, key=lambda o: (o.lower().find(q), len(o)))
    if sub:
        return sub[:limit]
    # fuzzy fallback
    return difflib.get_close_matches(query.strip(), options, n=limit, cutoff=0.4)


def country_to_continent(country_raw: str) -> str | None:
    c = (country_raw or "").strip()
    europe = {
        "United Kingdom", "Netherlands", "Sweden", "Denmark", "Norway", "Ireland",
        "Germany", "France", "Spain", "Belgium", "Italy", "Switzerland", "Austria", "Luxembourg"
    }
    na = {"United States", "Canada", "Mexico"}
    oceania = {"Australia", "New Zealand"}
    asia = {"Japan", "Singapore", "Hong Kong"}
    if c in europe: return "Europe"
    if c in na: return "North America"
    if c in oceania: return "Oceania"
    if c in asia: return "Asia"
    return None


def infer_raw_columns(mdl) -> list[str] | None:
    """
    Best-case: sklearn stored feature_names_in_ on the pipeline or its first transformer.
    That gives exact raw input columns (and order).
    """
    if hasattr(mdl, "feature_names_in_"):
        return list(mdl.feature_names_in_)

    if hasattr(mdl, "named_steps"):
        for step in mdl.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


def positive_class_index(classes):
    # Prefer label 1, else "successful", else last class.
    if classes is None:
        return -1
    cls = list(classes)
    if 1 in cls:
        return cls.index(1)
    for key in ["successful", "success", "funded", "yes", True]:
        if key in cls:
            return cls.index(key)
    return len(cls) - 1


# -----------------------------
# UI
# -----------------------------
st.caption("Backers is required by this model but is fixed internally (not shown).")

project_name = st.text_input("Project name (optional)", "")

col1, col2 = st.columns(2)
with col1:
    country_display = st.selectbox("Country", COUNTRIES_DISPLAY)
    category = st.selectbox("Category", CATEGORIES)

with col2:
    goal = st.number_input("Goal", min_value=0.0, value=10000.0, step=100.0)
    duration_days = st.number_input("Project duration (days)", min_value=1, max_value=365, value=30, step=1)

# Subcategory: type + suggestions
subcat_query = st.text_input("Subcategory (type to search)", "")
matches = suggest_subcats(subcat_query, SUBCATEGORIES, limit=10)

chosen_subcat = subcat_query.strip()
if matches:
    picked = st.selectbox("Suggestions", ["(use typed value)"] + matches)
    if picked != "(use typed value)":
        chosen_subcat = picked

# Map to raw training string for country (handles "Netherlands " case)
country_raw = DISPLAY_TO_RAW_COUNTRY.get(country_display, country_display)

# Values we can supply
value_map = {
    "Name": project_name,
    "Category": category,
    "Subcategory": chosen_subcat,
    "Country": country_raw,
    "Continent": country_to_continent(country_raw),
    "Goal": float(goal),
    "Project_Duration_Days": int(duration_days),
    "Backers": int(BACKERS_HIDDEN_DEFAULT),
    "Pledged": float(PLEDGED_HIDDEN_DEFAULT),
    "Goal_Fulfillment": float(GOAL_FULFILLMENT_DEFAULT),
}

raw_cols = infer_raw_columns(model)

# If we cannot infer, fall back to the 6 columns you stated for preprocessor3_adv.
if raw_cols is None:
    raw_cols = ["Goal", "Backers", "Project_Duration_Days", "Category", "Country", "Subcategory"]

X_raw = pd.DataFrame([{c: value_map.get(c, None) for c in raw_cols}], columns=raw_cols)


# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    # 1) FIRST TRY: model directly on raw input (works if model is a Pipeline with internal preprocessor)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_raw)[0]
            classes = getattr(model, "classes_", None)
            idx = positive_class_index(classes if classes is not None else list(range(len(proba))))
            p = float(proba[idx])

            st.subheader("Result")
            st.write("Prediction:", "✅ Funded" if p >= 0.5 else "❌ Not funded")
            st.write("Probability (funded):", f"{p:.2%}")
        else:
            pred = model.predict(X_raw)[0]
            st.subheader("Result")
            st.write("Predicted class:", pred)

    except Exception as e_direct:
        # 2) FALLBACK: external preprocessor + model (only correct if model is a bare estimator)
        try:
            if preprocessor is None:
                raise RuntimeError("No fallback preprocessor available.")

            Xt = preprocessor.transform(X_raw)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xt)[0]
                classes = getattr(model, "classes_", None)
                idx = positive_class_index(classes if classes is not None else list(range(len(proba))))
                p = float(proba[idx])

                st.subheader("Result")
                st.write("Prediction:", "✅ Funded" if p >= 0.5 else "❌ Not funded")
                st.write("Probability (funded):", f"{p:.2%}")
            else:
                pred = model.predict(Xt)[0]
                st.subheader("Result")
                st.write("Predicted class:", pred)

        except Exception as e_fallback:
            st.error("Prediction failed.")
            st.code(f"Direct model error:\n{e_direct}\n\nFallback error:\n{e_fallback}")

            with st.expander("Debug"):
                st.write("Model type:", type(model))
                if hasattr(model, "named_steps"):
                    st.write("Pipeline steps:", list(model.named_steps.keys()))
                st.write("Raw columns used:", raw_cols)
                st.dataframe(X_raw)

