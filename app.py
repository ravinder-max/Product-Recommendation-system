
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from html import escape

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. LOAD TRAINED MODEL ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    model_path = Path("models") / "recommendation_models.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "Model file not found. Run the model-saving cell "
            "in app.ipynb first."
        )

    return joblib.load(model_path)


try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load the trained models: {exc}")
    st.stop()

products = artifacts["products"].copy()
popularity = artifacts["popularity_model"].copy()
product_features = artifacts["product_features"].copy()
similarity = artifacts["similarity_matrix"]
product_index = artifacts["product_index"]
customer_to_idx = artifacts["customer_to_idx"]
product_ids = artifacts["product_ids"]
customer_factors = artifacts["customer_factors"]
product_factors = artifacts["product_factors"]
train_seen = artifacts["train_seen"]
evaluation = artifacts["evaluation_results"].copy()

# =========================================================
# 3. PREPARE PRODUCT CATALOGUE
# =========================================================
required_columns = [
    "product_id",
    "product_name",
    "product_category",
    "brand",
    "unit_price"
]

for col in required_columns:
    if col not in products.columns:
        products[col] = np.nan

products = products.drop_duplicates("product_id").copy()

products["product_name"] = (
    products["product_name"]
    .fillna("Unnamed product")
    .astype(str)
)

products["product_category"] = (
    products["product_category"]
    .fillna("Uncategorised")
    .astype(str)
)

products["brand"] = (
    products["brand"]
    .fillna("Unknown brand")
    .astype(str)
)

product_lookup = products.set_index("product_id", drop=False)

# Add product information to popularity data.
# The popularity model stores ranking statistics, not product names.
product_details = products[
    [
        "product_id",
        "product_name",
        "product_category",
        "brand",
        "unit_price"
    ]
].copy()

popularity = popularity.drop(
    columns=[
        "product_name",
        "product_category",
        "brand",
        "unit_price"
    ],
    errors="ignore"
)

popularity = popularity.merge(
    product_details,
    on="product_id",
    how="left",
    validate="many_to_one"
)

# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================
def get_product_details(product_id):
    """Return one product's catalogue details."""
    if product_id not in product_lookup.index:
        return None

    row = product_lookup.loc[product_id]

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    return row


def display_products(df):
    """Display products in a visually defined two-column card layout."""
    if df is None or df.empty:
        st.info("No recommendations are available for this selection.")
        return

    st.markdown(
        """
        <style>
        .product-card {
            border: 1px solid #dfe6ee;
            border-radius: 0.85rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            padding: 1rem 1rem 0.75rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
            height: 100%;
        }
        .product-card-header {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 0.75rem;
        }
        .product-rank {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 1.8rem;
            height: 1.8rem;
            border-radius: 50%;
            background: #e8f0ff;
            color: #1d4ed8;
            font-weight: 700;
            font-size: 0.8rem;
        }
        .product-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.4;
        }
        .product-card-meta {
            margin: 0.2rem 0;
            font-size: 0.92rem;
            color: #334155;
        }
        .product-card-score {
            margin-top: 0.6rem;
            font-size: 0.8rem;
            color: #0f766e;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2)

    for i, (_, row) in enumerate(df.iterrows()):
        with columns[i % 2]:
            product_id = row.get("product_id", None)
            details = get_product_details(product_id)

            name = row.get("product_name", None)
            category = row.get("product_category", None)
            brand = row.get("brand", None)
            price = row.get("unit_price", np.nan)

            if details is not None:
                if pd.isna(name):
                    name = details["product_name"]
                if pd.isna(category):
                    category = details["product_category"]
                if pd.isna(brand):
                    brand = details["brand"]
                if pd.isna(price):
                    price = details["unit_price"]

            if name is None or pd.isna(name):
                name = "Unnamed product"
            if category is None or pd.isna(category):
                category = "Uncategorised"
            if brand is None or pd.isna(brand):
                brand = "Unknown brand"

            score = row.get(
                "Score",
                row.get("recommendation_score", np.nan)
            )

            safe_name = escape(str(name))
            safe_category = escape(str(category))
            safe_brand = escape(str(brand))
            safe_product_id = escape(str(product_id)) if product_id is not None else "N/A"
            price_html = ""
            if pd.notna(price):
                price_html = (
                    f'<div class="product-card-meta"><b>Price:</b> '
                    f'₹{float(price):,.2f}</div>'
                )

            score_html = ""
            if pd.notna(score):
                score_html = (
                    f'<div class="product-card-score"><b>Recommendation score:</b> '
                    f'{float(score):.4f}</div>'
                )

            st.markdown(
                f"""
                <div class="product-card">
                    <div class="product-card-header">
                        <span class="product-rank">{i + 1}</span>
                        <span class="product-card-title">{safe_name}</span>
                    </div>
                    <div class="product-card-meta"><b>Product ID:</b> {safe_product_id}</div>
                    <div class="product-card-meta"><b>Category:</b> {safe_category}</div>
                    <div class="product-card-meta"><b>Brand:</b> {safe_brand}</div>
                    {price_html}
                    {score_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_visual_insights():
    """Render a diverse set of charts for sales, pricing, ratings, and purchase trends."""
    st.title("📈 Visual Insights")
    st.write(
        "Explore product pricing, review sentiment, purchase volume, and similarity "
        "patterns across the catalog."
    )

    products_for_charts = products.copy()
    products_for_charts["unit_price"] = pd.to_numeric(
        products_for_charts["unit_price"], errors="coerce"
    )
    products_for_charts["product_rating"] = pd.to_numeric(
        products_for_charts["product_rating"], errors="coerce"
    )

    price_bins = pd.cut(
        products_for_charts["unit_price"].dropna(),
        bins=8,
        include_lowest=True
    )
    price_hist = (
        pd.DataFrame({"Price Range": price_bins})
        .groupby("Price Range")
        .size()
        .rename("Products")
        .reset_index()
    )
    price_hist["Price Range"] = price_hist["Price Range"].astype(str)

    category_price = (
        products_for_charts.groupby("product_category", dropna=False)["unit_price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    category_price.columns = ["Category", "Average Price"]

    category_rating = (
        products_for_charts.groupby("product_category", dropna=False)["product_rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    category_rating.columns = ["Category", "Average Rating"]

    brand_rating = (
        products_for_charts.groupby("brand", dropna=False)["product_rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    brand_rating.columns = ["Brand", "Average Rating"]

    purchase_summary = popularity.copy()
    detail_columns = [
        column for column in ["product_name", "product_category"]
        if column not in purchase_summary.columns
    ]
    if detail_columns:
        purchase_summary = purchase_summary.merge(
            products_for_charts[["product_id"] + detail_columns],
            on="product_id",
            how="left"
        )

    if "product_name" not in purchase_summary.columns:
        purchase_summary["product_name"] = "Unknown product"

    purchase_summary = purchase_summary.sort_values(
        "total_quantity",
        ascending=False
    ).head(12)

    price_rating = products_for_charts[["unit_price", "product_rating", "product_category"]].dropna()
    price_rating = price_rating.sample(n=min(len(price_rating), 800), random_state=42)

    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.subheader("1. Price Distribution")
        st.bar_chart(price_hist.set_index("Price Range"))

    with chart_cols[1]:
        st.subheader("2. Average Price by Category")
        st.bar_chart(category_price.set_index("Category"))

    with chart_cols[0]:
        st.subheader("3. Review Score by Category")
        st.line_chart(category_rating.set_index("Category"))

    with chart_cols[1]:
        st.subheader("4. Price vs Rating")
        st.scatter_chart(price_rating[["unit_price", "product_rating"]])

    st.subheader("5. Top Product Purchases")
    st.area_chart(
        purchase_summary.set_index("product_name")["total_quantity"]
    )

    st.subheader("6. Top Brands by Review")
    st.bar_chart(brand_rating.set_index("Brand"))

# =========================================================
# 5. MODEL 1: POPULARITY-BASED RECOMMENDATION
# =========================================================
def popular_recommendations(customer_id=None, k=10):
    seen = train_seen.get(customer_id, set()) if customer_id else set()

    results = popularity[
        ~popularity["product_id"].isin(seen)
    ].head(k).copy()

    return results.reset_index(drop=True)


# =========================================================
# 6. MODEL 2: CONTENT-BASED RECOMMENDATION
# =========================================================
def content_recommendations(product_id, customer_id=None, k=10):
    if product_id not in product_index.index:
        return pd.DataFrame()

    idx = int(product_index.loc[product_id])
    scores = similarity[idx]

    seen = train_seen.get(customer_id, set()) if customer_id else set()

    ranked_indices = np.argsort(scores)[::-1]

    selected_indices = []

    for i in ranked_indices:
        candidate_id = product_features.iloc[i]["product_id"]

        if candidate_id == product_id:
            continue

        if candidate_id in seen:
            continue

        selected_indices.append(i)

        if len(selected_indices) == k:
            break

    if not selected_indices:
        return pd.DataFrame()

    result = product_features.iloc[selected_indices][
        [
            "product_id",
            "product_name",
            "product_category",
            "brand"
        ]
    ].copy()

    result["Score"] = [
        float(scores[i]) for i in selected_indices
    ]

    # Attach price from the catalogue
    result = result.merge(
        products[["product_id", "unit_price"]].drop_duplicates(
            "product_id"
        ),
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    return result.reset_index(drop=True)


# =========================================================
# 7. MODEL 3: COLLABORATIVE FILTERING
# =========================================================
def collaborative_recommendations(customer_id, k=10):
    if customer_id not in customer_to_idx:
        return pd.DataFrame()

    customer_idx = customer_to_idx[customer_id]

    scores = customer_factors[customer_idx] @ product_factors.T

    seen = train_seen.get(customer_id, set())
    ranked_indices = np.argsort(scores)[::-1]

    recommended_ids = []

    for idx in ranked_indices:
        pid = product_ids[idx]

        if pid in seen:
            continue

        recommended_ids.append(pid)

        if len(recommended_ids) == k:
            break

    if not recommended_ids:
        return pd.DataFrame()

    result = products[
        products["product_id"].isin(recommended_ids)
    ].copy()

    # Preserve model ranking order
    rank_map = {
        pid: rank
        for rank, pid in enumerate(recommended_ids)
    }

    result["_rank"] = result["product_id"].map(rank_map)
    result = result.sort_values("_rank").drop(columns="_rank")

    result["Score"] = [
        float(scores[np.where(product_ids == pid)[0][0]])
        for pid in result["product_id"]
    ]

    return result[
        [
            "product_id",
            "product_name",
            "product_category",
            "brand",
            "unit_price",
            "Score"
        ]
    ].reset_index(drop=True)


# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🛍️ Product AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Visual Insights",
        "Product Explorer",
        "Recommendations",
        "Model Comparison",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("ML Product Recommendation Project")

# =========================================================
# 9. DASHBOARD
# =========================================================
if page == "Dashboard":
    st.title("🛍️ Product Recommendation System")
    st.write(
        "Explore products and compare three recommendation "
        "approaches using historical purchase data."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Products",
        f"{products['product_id'].nunique():,}"
    )

    c2.metric(
        "Training customers",
        f"{len(customer_to_idx):,}"
    )

    c3.metric(
        "Recommendation models",
        "3"
    )

    st.subheader("Top Popular Products")
    display_products(popular_recommendations(k=10))

    st.subheader("Model Evaluation Summary")
    st.dataframe(
        evaluation,
        use_container_width=True,
        hide_index=True
    )

elif page == "Visual Insights":
    render_visual_insights()

# =========================================================
# 10. PRODUCT EXPLORER
# =========================================================
elif page == "Product Explorer":
    st.title("🔎 Product Explorer")

    c1, c2 = st.columns(2)

    categories = sorted(
        products["product_category"].dropna().unique().tolist()
    )

    brands = sorted(
        products["brand"].dropna().unique().tolist()
    )

    with c1:
        selected_category = st.selectbox(
            "Category",
            ["All categories"] + categories
        )

    with c2:
        selected_brand = st.selectbox(
            "Brand",
            ["All brands"] + brands
        )

    search = st.text_input("Search by product name")

    filtered = products.copy()

    if selected_category != "All categories":
        filtered = filtered[
            filtered["product_category"] == selected_category
        ]

    if selected_brand != "All brands":
        filtered = filtered[
            filtered["brand"] == selected_brand
        ]

    if search.strip():
        filtered = filtered[
            filtered["product_name"].str.contains(
                search.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    st.write(f"**{len(filtered)} products found**")

    display_products(filtered.head(30))


# =========================================================
# 11. RECOMMENDATIONS PAGE
# =========================================================
elif page == "Recommendations":
    st.title("🎯 Product Recommendations")

    mode = st.radio(
        "Choose recommendation mode",
        [
            "Personalised by customer",
            "Similar to a product"
        ]
    )

    k = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    if mode == "Personalised by customer":
        customer_options = list(customer_to_idx.keys())

        customer_id = st.selectbox(
            "Select customer",
            customer_options
        )

        seed_options = products["product_id"].tolist()

        seed_id = st.selectbox(
            "Seed product for content-based model",
            seed_options,
            format_func=lambda pid: (
                f"{pid} — "
                f"{get_product_details(pid)['product_name']}"
            )
        )

        if st.button("Generate recommendations"):
            st.subheader("Model 1 — Popularity-Based")
            display_products(
                popular_recommendations(customer_id, k)
            )

            st.subheader("Model 2 — Content-Based TF-IDF")
            display_products(
                content_recommendations(seed_id, customer_id, k)
            )

            st.subheader("Model 3 — Collaborative Filtering")
            display_products(
                collaborative_recommendations(customer_id, k)
            )

    else:
        seed_options = products["product_id"].tolist()

        seed_id = st.selectbox(
            "Choose a product",
            seed_options,
            format_func=lambda pid: (
                f"{pid} — "
                f"{get_product_details(pid)['product_name']}"
            )
        )

        if st.button("Find similar products"):
            display_products(
                content_recommendations(seed_id, k=k)
            )


# =========================================================
# 12. MODEL COMPARISON PAGE
# =========================================================
elif page == "Model Comparison":
    st.title("📊 Model Performance Comparison")

    st.write(
        "These results are from the offline evaluation performed "
        "in your notebook. Higher Precision, Recall and NDCG "
        "indicate better ranking performance."
    )

    st.dataframe(
        evaluation.sort_values(
            "NDCG@10",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    for metric in [
        "Precision@10",
        "Recall@10",
        "NDCG@10"
    ]:
        if metric in evaluation.columns:
            st.subheader(metric)

            chart = evaluation.set_index("Model")[[metric]]
            st.bar_chart(chart)

    coverage_column = "Catalog Coverage (%)"

    if coverage_column in evaluation.columns:
        st.subheader("Catalogue Coverage")

        st.bar_chart(
            evaluation.set_index("Model")[[coverage_column]]
        )

    st.info(
        "Content-Based TF-IDF achieved the highest measured "
        "scores in your current evaluation, but the differences "
        "are small. Further experiments may change the ranking."
    )

# =========================================================
# 13. ABOUT PAGE
# =========================================================
elif page == "About":
    st.title("ℹ️ About")

    st.markdown(
        """
        <div style="padding:1.35rem 1.5rem; border:1px solid #cbd5e1; border-radius:16px; background:linear-gradient(180deg, #0f172a 0%, #1e293b 100%); box-shadow:0 8px 24px rgba(15,23,42,0.12);">
            <h3 style="margin:0 0 0.45rem 0; color:#f8fafc; font-size:1.5rem;">Made by Ravinder Singh</h3>
            <p style="margin:0; color:#e2e8f0; font-size:1rem; line-height:1.6;">Bio: (ADD Space)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    about_style = """
    <style>
    .about-panel {
        background: #f8fafc;
        border: 1px solid #dfe7f1;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }
    .about-panel h4 {
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .about-panel p, .about-panel li {
        color: #334155;
        font-size: 0.98rem;
        line-height: 1.7;
    }
    .about-panel strong {
        color: #0f172a;
    }
    </style>
    """
    st.markdown(about_style, unsafe_allow_html=True)

    tech_cols = st.columns(2)
    with tech_cols[0]:
        st.markdown(
            """
            <div class="about-panel">
                <h4>Tech Used</h4>
                <ul>
                    <li>Python</li>
                    <li>Streamlit</li>
                    <li>Pandas</li>
                    <li>NumPy</li>
                    <li>Scikit-learn</li>
                    <li>Joblib</li>
                    <li>Jupyter Notebook</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech_cols[1]:
        st.markdown(
            """
            <div class="about-panel">
                <h4>Project Overview</h4>
                <p>This product recommendation system evaluates multiple recommendation strategies on e-commerce data and presents them in a simple dashboard for business exploration and comparison.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="about-panel" style="margin-top: 1.5rem;">
            <h4>Model Descriptions</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_info = {
        "Popularity-Based": "Ranks products by overall sales and popularity. Best for quick, simple recommendations and baseline performance.",
        "Content-Based TF-IDF": "Uses product attributes and text similarity to recommend items that are similar to a selected product or customer preference.",
        "Collaborative Filtering": "Learns user-item patterns from historical purchases and recommends products that similar users tend to buy."
    }

    for model_name, description in model_info.items():
        st.markdown(
            f"<div class='about-panel'><strong>{model_name}:</strong> {description}</div>",
            unsafe_allow_html=True,
        )

# =========================================================
# 14. FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Product Recommendation System | Built with Python, "
    "Pandas, Scikit-learn and Streamlit"
)
