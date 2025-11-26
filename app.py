import os
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import onnxruntime as ort
import urllib.parse
import pickle

# -------------------------------------------------------
# UI CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Mood Music Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIMPLE DARK THEME
st.markdown(
    """
    <style>
    body { background-color: #0f1117; color: white; }
    .main { background-color: #0f1117; }
    .play-btn {
        background-color: #1db954;
        padding: 8px 14px;
        border-radius: 8px;
        color: white;
        text-decoration: none;
        font-size: 13px;
        margin-right: 8px;
    }
    .yt-btn {
        background-color: #ff0000;
        padding: 8px 14px;
        border-radius: 8px;
        color: white;
        text-decoration: none;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------
MODEL_ONNX_PATH = "emotion_cnn.onnx"
DATA_PATH = "Hindi_songs_clean.csv"
POSTER_IMAGE = "songs_cover.jpg"
SIM_PATH = "similarity.npy"
IDX_PATH = "index_mapping.pkl"

# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

df = load_dataset()

if df is None:
    st.error("Missing dataset: Hindi_songs_clean.csv")
    st.stop()

if 'mood' not in df.columns:
    st.error("Dataset missing 'mood' column.")
    st.stop()

# -------------------------------------------------------
# LOAD ONNX MODEL
# -------------------------------------------------------
session = ort.InferenceSession(MODEL_ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# -------------------------------------------------------
# LOAD CF ARTIFACTS
# -------------------------------------------------------
@st.cache_resource
def load_cf():
    if not os.path.exists(SIM_PATH) or not os.path.exists(IDX_PATH):
        return None, None
    similarity = np.load(SIM_PATH, mmap_mode="r")
    with open(IDX_PATH, "rb") as f:
        index_mapping = pickle.load(f)
    return similarity, index_mapping

similarity, index_mapping = load_cf()

# -------------------------------------------------------
# EMOTION + MOOD MAP
# -------------------------------------------------------
EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']
MOOD_MAP = {
    "angry": "Energetic",
    "disgust": "Sad",
    "fear": "Calm",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Calm"
}
MOOD_EMOJI = {
    "Happy": "🙂",
    "Energetic": "🔥",
    "Sad": "😢",
    "Calm": "😌",
    "Surprise": "😲"
}

# -------------------------------------------------------
# FACE PROCESSING
# -------------------------------------------------------
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

def preprocess_face(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48,48)) / 255.0
    return face.reshape(1,48,48,1).astype(np.float32)

def predict_emotion(img48):
    preds = session.run(None, {input_name: img48})[0][0]

    # Soft penalty for surprise (not too strong)
    preds[EMOTIONS.index('surprise')] *= 0.85

    # Get top two predictions
    sorted_idx = preds.argsort()[::-1]
    top1 = sorted_idx[0]
    top2 = sorted_idx[1]

    top1_prob = preds[top1]
    top2_prob = preds[top2]

    # 1️⃣ Very low confidence → show "Unknown" instead of Neutral/Calm
    if top1_prob < 0.40:
        return "unknown"

    # 2️⃣ If Surprise barely wins → take 2nd best
    if EMOTIONS[top1] == "surprise" and (top1_prob - top2_prob) < 0.10:
        return EMOTIONS[top2]

    # 3️⃣ Normal prediction
    return EMOTIONS[top1]



# -------------------------------------------------------
# MOOD PLAYLIST
# -------------------------------------------------------
def mood_recommend(mood, top_n=10):
    mood_df = df[df['mood'].str.lower() == mood.lower()]
    if mood_df.empty:
        return mood_df
    return mood_df.sample(min(top_n, len(mood_df)))

# -------------------------------------------------------
# COLLABORATIVE FILTERING RECOMMENDER
# -------------------------------------------------------
def recommend_cf(seed_song, top_n=10):
    if similarity is None or index_mapping is None:
        return pd.DataFrame()

    # Normalize
    seed_norm = seed_song.strip().lower()
    mapped = {k.strip().lower(): v for k, v in index_mapping["song_to_index"].items()}

    if seed_norm not in mapped:
        return pd.DataFrame()

    idx = mapped[seed_norm]
    sims = similarity[idx]

    similar_indices = sims.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    inv_map = {v: k for k, v in index_mapping["song_to_index"].items()}
    songs = [inv_map[i] for i in similar_indices]

    return df[df['song_name'].str.lower().isin([s.lower() for s in songs])].reset_index(drop=True)

# -------------------------------------------------------
# SONG CARD
# -------------------------------------------------------
def render_song_card(row):
    song = row['song_name']
    artist = row['singer']
    mood = row['mood']
    emoji = MOOD_EMOJI.get(mood, "🎵")

    spotify_url = "https://open.spotify.com/search/" + urllib.parse.quote(song + " " + artist)
    youtube_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(song + " " + artist)

    with st.container():
        st.image(POSTER_IMAGE, use_column_width=True)
        st.markdown(f"### {emoji} {song}")
        st.markdown(f"**{artist}**")
        st.markdown(f"**Mood:** {mood}")
        st.markdown(
            f"""
            <a href="{spotify_url}" target="_blank" class="play-btn">Spotify ▶️</a>
            <a href="{youtube_url}" target="_blank" class="yt-btn">YouTube ▶️</a>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.title("Menu")
page = st.sidebar.radio("", ["Mood Playlist", "Search Songs"])

# -------------------------------------------------------
# PAGE 1: Mood Playlist
# -------------------------------------------------------
if page == "Mood Playlist":
    st.header("📸 Capture Mood")
    cam = st.camera_input("Capture your face")

    if cam and st.button("Analyze Mood"):
        face = preprocess_face(cam.getvalue())
        emotion = predict_emotion(face)
        mood = MOOD_MAP.get(emotion, "Calm")
        emoji = MOOD_EMOJI.get(mood, "🎵")

        st.markdown(f"<h2>{emoji} Detected Mood: {mood}</h2>", unsafe_allow_html=True)

        recs = mood_recommend(mood, 10)

        cols = st.columns(2)
        idx = 0
        for _, row in recs.iterrows():
            with cols[idx % 2]:
                render_song_card(row)
            idx += 1

# -------------------------------------------------------
# PAGE 2: ADVANCED SEARCH + CF
# -------------------------------------------------------
else:
    st.header("🔍 Advanced Song Search")

    # --- Search Inputs ---
    name_query = st.text_input("Search by Song Name")

    artist_list = sorted(df['singer'].dropna().unique().tolist())
    artist_filter = st.selectbox("Filter by Artist", ["All"] + artist_list)

    mood_list = sorted(df['mood'].dropna().unique().tolist())
    mood_filter = st.selectbox("Filter by Mood", ["All"] + mood_list)

    # Optional year filter
    if 'year' in df.columns:
        year_list = sorted(df['year'].dropna().unique().tolist())
        year_filter = st.selectbox("Filter by Year", ["All"] + list(map(str, year_list)))
    else:
        year_filter = "All"

    # --- Apply Filters ---
    filtered = df.copy()

    if name_query:
        filtered = filtered[filtered['song_name'].str.contains(name_query, case=False, na=False)]

    if artist_filter != "All":
        filtered = filtered[filtered['singer'] == artist_filter]

    if mood_filter != "All":
        filtered = filtered[filtered['mood'] == mood_filter]

    if year_filter != "All":
        filtered = filtered[filtered['year'].astype(str) == year_filter]

    # --- Display Results ---
    if filtered.empty:
        st.warning("No songs found with selected filters.")
    else:
        st.write(f"### Results: {len(filtered)} songs found")

        cols = st.columns(2)
        idx = 0
        for _, row in filtered.head(20).iterrows():
            with cols[idx % 2]:
                render_song_card(row)

                # CF Button
                if st.button(f"Find Similar to {row['song_name']}", key=row['song_name']):
                    st.subheader(f"🔥 Similar Songs to '{row['song_name']}'")
                    recs = recommend_cf(row['song_name'], 10)

                    if recs.empty:
                        st.warning("No similar songs found.")
                    else:
                        sub_cols = st.columns(2)
                        sub_idx = 0
                        for _, rec_row in recs.iterrows():
                            with sub_cols[sub_idx % 2]:
                                render_song_card(rec_row)
                            sub_idx += 1
            idx += 1
