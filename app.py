from pathlib import Path
import joblib
import streamlit as st

BASE = Path(__file__).resolve().parent
PRE_PATH = BASE / "models" / "preprocessor3_adv.joblib"
MODEL_PATH = BASE / "models" / "model_lr_no_subcat.joblib"

@st.cache_resource
def load_artifacts():
    return joblib.load(PRE_PATH), joblib.load(MODEL_PATH)

preprocessor, model = load_artifacts()


def unique_clean(seq):
    seen = set()
    out = []
    for x in seq:
        if x is None:
            continue
        x2 = str(x).strip()
        if x2 and x2 not in seen:
            seen.add(x2)
            out.append(x2)
    return out

CATEGORIES = unique_clean([
    "Fashion", "Film & Video", "Art", "Technology", "Journalism",
    "Publishing", "Theater", "Music", "Photography", "Games", "Design",
    "Food", "Crafts", "Comics", "Dance"
])

COUNTRIES = unique_clean([
    "United States", "United Kingdom", "Canada", "Australia",
    "New Zealand", "Netherlands ", "Sweden", "Denmark", "Norway",
    "Ireland", "Germany", "France", "Spain", "Belgium", "Italy",
    "Switzerland", "Austria", "Luxembourg", "Singapore", "Hong Kong",
    "Mexico", "Japan"
])

SUBCATEGORIES = unique_clean([
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
])

# ---- UI inputs ----
name = st.text_input("Project name (not used by model)", "")

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", COUNTRIES)
    category = st.selectbox("Category", CATEGORIES)
with col2:
    subcategory = st.selectbox("Subcategory", SUBCATEGORIES)
    goal = st.number_input("Goal", min_value=0.0, value=10000.0, step=100.0)

threshold = st.slider("Decision threshold (only used if probability exists)", 0.05, 0.95, 0.50, 0.01)

# Build the exact columns expected by your preprocessor
X = pd.DataFrame([{
    "Goal": goal,
    "Country": country,
    "Category": category,
}])

if st.button("Predict"):
    try:
        Xt = preprocessor.transform(X)

        # Predict class
        pred = model.predict(Xt)[0]

        st.subheader(f"Result for: {name or '(no name)'}")

        # If model provides probabilities, show them and apply threshold.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xt)[0]
            classes = getattr(model, "classes_", [0, 1])

            # Heuristic: treat the "1" class as funded if present, otherwise take last class.
            if 1 in classes:
                funded_idx = list(classes).index(1)
            else:
                funded_idx = len(classes) - 1

            p_funded = float(proba[funded_idx])
            funded = p_funded >= threshold

            st.write("Classes order:", classes)
            st.write("Probability (assumed funded class):", f"{p_funded:.2%}")
            st.write("Prediction (by threshold):", "✅ Funded" if funded else "❌ Not funded")
        else:
            # No probability available; just show raw predicted class
            st.write("Predicted class:", pred)

    except Exception as e:
        st.error("Prediction failed. Most common cause: column mismatch or unexpected category value.")
        st.code(str(e))
