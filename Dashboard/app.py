import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Phone Market Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    df = pd.read_csv("analysis_master.csv", dtype=str, low_memory=False)
    df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce")
    df["date_vn"] = pd.to_datetime(df["date_vn"], errors="coerce")
    return df

df = load_data()

st.title("📱 Smartphone Market Analysis Dashboard")

# Sidebar filter
brands = ["All"] + sorted(df["brand"].dropna().unique().tolist())
selected_brand = st.sidebar.selectbox("Chọn thương hiệu", brands)

if selected_brand != "All":
    df = df[df["brand"] == selected_brand]

# ===== TAB SETUP =====
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🙂 Sentiment Analysis", "🧩 Topic Analysis", "🏷 Brand × Topic", "🔍 Search Engine"]
)

# =====================================================================================
# TAB 1 – OVERVIEW
# =====================================================================================
with tab1:
    st.header("📊 Tổng quan dữ liệu")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng bình luận", len(df))
    col2.metric("Số topic", df["topic_id"].nunique())
    col3.metric("Các thương hiệu", df["brand"].nunique())

    st.subheader("Phân bố sentiment")
    plt.figure(figsize=(6,3))
    sns.countplot(data=df, x="weak_label", palette="Set2")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Phân bố bình luận theo thương hiệu")
    plt.figure(figsize=(6,3))
    sns.countplot(data=df, x="brand", palette="viridis")
    plt.xticks(rotation=20)
    st.pyplot(plt.gcf())
    plt.clf()

# =====================================================================================
# TAB 2 – SENTIMENT ANALYSIS
# =====================================================================================
with tab2:
    st.header("🙂 Sentiment Analysis")

    st.subheader("Tần suất sentiment theo ngày")
    df_daily = df.dropna(subset=["date_vn"]).groupby(["date_vn","weak_label"]).size().reset_index(name="count")

    plt.figure(figsize=(12,5))
    sns.lineplot(data=df_daily, x="date_vn", y="count", hue="weak_label")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Sentiment theo brand")
    plt.figure(figsize=(10,4))
    sns.countplot(data=df, x="brand", hue="weak_label", palette="Set2")
    plt.xticks(rotation=20)
    st.pyplot(plt.gcf())
    plt.clf()

# =====================================================================================
# TAB 3 – TOPIC ANALYSIS
# =====================================================================================
with tab3:
    st.header("🧩 Topic Analysis")

    # Top topics
    st.subheader("Top 20 Topic")
    top_topics = df["topic_id"].value_counts().head(20)

    plt.figure(figsize=(10,5))
    sns.barplot(x=top_topics.values, y=top_topics.index, palette="viridis")
    plt.xlabel("Count")
    plt.ylabel("Topic ID")
    st.pyplot(plt.gcf())
    plt.clf()

    # Topic detail
    selected_topic = st.selectbox("Xem chi tiết topic", top_topics.index)

    df_topic_sample = df[df["topic_id"] == selected_topic].head(10)
    st.write("🔎 **10 representative comments**:")
    st.table(df_topic_sample[["text_clean","brand","weak_label"]])

# =====================================================================================
# TAB 4 – BRAND × TOPIC HEATMAP
# =====================================================================================
with tab4:
    st.header("🏷 Brand × Topic Heatmap")

    df_no_noise = df[df["topic_id"] != -1]
    top20_ids = df_no_noise["topic_id"].value_counts().head(20).index
    df_top20 = df_no_noise[df_no_noise["topic_id"].isin(top20_ids)]

    topic_brand = pd.crosstab(df_top20["topic_id"], df_top20["brand"])

    plt.figure(figsize=(10,7))
    sns.heatmap(topic_brand, cmap="YlGnBu")
    st.pyplot(plt.gcf())
    plt.clf()

    st.caption("Lưu ý: Topic -1 (noise) đã được loại bỏ.")

# =====================================================================================
# TAB 5 – SEARCH ENGINE
# =====================================================================================
with tab5:
    st.header("🔍 Search comments")

    query = st.text_input("Nhập từ khoá để tìm bình luận:")

    if query:
        df_search = df[df["text_clean"].str.contains(query, case=False, na=False)]
        st.write(f"🔎 Tìm thấy **{len(df_search)}** bình luận chứa từ khóa '{query}'")

        st.dataframe(df_search[["text_clean","brand","weak_label","topic_id"]].head(50),
                     use_container_width=True)
