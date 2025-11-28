import os
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import onnxruntime as ort
import urllib.parse
import pickle

# -------------------------------------------------------
# 1. UI CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Mood Music Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme & Button Styles
st.markdown(
    """
    <style>
    body { background-color: #0f1117; color: white; }
    .main { background-color: #0f1117; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .play-btn {
        background-color: #1db954;
        padding: 8px 14px;
        border-radius: 8px;
        color: white;
        text-decoration: none;
        font-size: 13px;
        margin-right: 8px;
        display: inline-block;
    }
    .yt-btn {
        background-color: #ff0000;
        padding: 8px 14px;
        border-radius: 8px;
        color: white;
        text-decoration: none;
        font-size: 13px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# 2. CONSTANTS & PATHS
# -------------------------------------------------------
MODEL_ONNX_PATH = "emotion_cnn.onnx"
DATA_PATH = "Hindi_songs_clean.csv"
POSTER_IMAGE = "songs_cover.jpg"
SIM_PATH = "similarity.npy"
IDX_PATH = "index_mapping.pkl"

# Haar Cascade for Face Detection
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# -------------------------------------------------------
# 3. DATA LOADING
# -------------------------------------------------------
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

df = load_dataset()

if df is None:
    st.error(f"❌ Missing dataset: {DATA_PATH}")
    st.stop()

if 'mood' not in df.columns:
    st.error("❌ Dataset is missing the 'mood' column.")
    st.stop()

# -------------------------------------------------------
# 4. MODEL LOADING
# -------------------------------------------------------
# Load ONNX Model
try:
    session = ort.InferenceSession(MODEL_ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
except Exception as e:
    st.error(f"❌ Error loading ONNX model: {e}")
    st.stop()

# Load Face Detector
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

# Load Collaborative Filtering Artifacts
@st.cache_resource
def load_cf():
    if not os.path.exists(SIM_PATH) or not os.path.exists(IDX_PATH):
        return None, None
    try:
        similarity = np.load(SIM_PATH, mmap_mode="r")
        with open(IDX_PATH, "rb") as f:
            index_mapping = pickle.load(f)
        return similarity, index_mapping
    except Exception:
        return None, None

similarity, index_mapping = load_cf()

# -------------------------------------------------------
# 5. MAPPINGS
# -------------------------------------------------------
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

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
# 6. CORE FUNCTIONS (PREPROCESS & PREDICT)
# -------------------------------------------------------

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_face(img_bytes):
    """
    Decodes image, finds face, crops it, and resizes to 48x48.
    Returns None if no face is found.
    """
    try:
        # Decode
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect Faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # If no face, return None
        if len(faces) == 0:
            return None

        # Crop Largest Face
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        
        # Resize & Normalize
        resized = cv2.resize(face_img, (48, 48))
        normalized = resized / 255.0
        
        return normalized.reshape(1, 48, 48, 1).astype(np.float32)

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def predict_emotion(img48):
    """Runs inference on the cropped 48x48 face. Always returns a result."""
    # Run the model
    preds = session.run(None, {input_name: img48})[0][0]
    
    # Convert Logits to Probability (Softmax)
    probs = softmax(preds)

    # Soft penalty for surprise (optional heuristic to reduce false positives)
    probs[EMOTIONS.index('surprise')] *= 0.85

    # Get top prediction
    sorted_idx = probs.argsort()[::-1]
    top1 = sorted_idx[0]
    
    # We always return the top emotion, even if confidence is low.
    prediction = EMOTIONS[top1]
    score = probs[top1]
    
    return prediction, score, probs

# -------------------------------------------------------
# 7. RECOMMENDATION LOGIC
# -------------------------------------------------------

def mood_recommend(mood, top_n=10):
    mood_df = df[df['mood'].str.lower() == mood.lower()]
    if mood_df.empty:
        return mood_df
    return mood_df.sample(min(top_n, len(mood_df)))

def recommend_cf(seed_song, top_n=10):
    if similarity is None or index_mapping is None:
        return pd.DataFrame()

    seed_norm = seed_song.strip().lower()
    mapped = {k.strip().lower(): v for k, v in index_mapping["song_to_index"].items()}

    if seed_norm not in mapped:
        return pd.DataFrame()

    idx = mapped[seed_norm]
    sims = similarity[idx]

    similar_indices = sims.argsort()[::-1]
    # Skip the first one (it's the song itself)
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    inv_map = {v: k for k, v in index_mapping["song_to_index"].items()}
    songs = [inv_map[i] for i in similar_indices]

    return df[df['song_name'].str.lower().isin([s.lower() for s in songs])].reset_index(drop=True)

def render_song_card(row):
    song = row['song_name']
    artist = row['singer']
    mood = row['mood']
    emoji = MOOD_EMOJI.get(mood, "🎵")

    spotify_url = "https://open.spotify.com/search/" + urllib.parse.quote(song + " " + artist)
    youtube_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(song + " " + artist)

    with st.container():
        if os.path.exists(POSTER_IMAGE):
            st.image(POSTER_IMAGE, use_column_width=True)
        else:
            st.write("🎵") # Fallback if image missing
            
        st.markdown(f"### {emoji} {song}")
        st.markdown(f"**{artist}**")
        st.caption(f"Mood: {mood}")
        st.markdown(
            f"""
            <a href="{spotify_url}" target="_blank" class="play-btn">Spotify</a>
            <a href="{youtube_url}" target="_blank" class="yt-btn">YouTube</a>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------------
# 8. MAIN APP NAVIGATION
# -------------------------------------------------------
st.sidebar.title("🎵 Mood Tuner")
page = st.sidebar.radio("Navigate", ["Mood Playlist (Camera)", "Advanced Search"])

# --- PAGE 1: CAMERA & MOOD ---
if page == "Mood Playlist (Camera)":
    st.header("📸 How are you feeling?")
    st.write("Take a selfie, and we'll play music that matches your mood.")
    
    cam = st.camera_input("Look into the camera")

    if cam:
        if st.button("Analyze Mood", type="primary"):
            with st.spinner("Analyzing your face..."):
                # 1. Preprocess (Crop Face)
                face = preprocess_face(cam.getvalue())

                # 2. Check if face found
                if face is None:
                    st.error("⚠️ No face detected! Please move closer or adjust lighting.")
                else:
                    # 3. Predict (Now returns score and all probs)
                    emotion, score, probs = predict_emotion(face)
                    
                    # 4. Map to Mood
                    mood = MOOD_MAP.get(emotion, "Calm")
                    emoji = MOOD_EMOJI.get(mood, "🎵")

                    # Display Result
                    st.success(f"Detected: **{emotion.title()}** ({int(score*100)}% confidence)")
                    st.markdown(f"<h1 style='text-align:center; font-size:60px;'>{emoji} {mood}</h1>", unsafe_allow_html=True)

                    # --- SHOW DEBUG INFO ---
                    with st.expander("📊 See Emotion Details (Debug Info)"):
                        st.write("Model Probabilities:")
                        # Create a nice dataframe for the chart
                        chart_data = pd.DataFrame(
                            {"Emotion": EMOTIONS, "Probability": probs}
                        ).set_index("Emotion")
                        st.bar_chart(chart_data)

                    # 5. Recommend Songs
                    recs = mood_recommend(mood, 8)
                    
                    st.markdown("---")
                    if recs.empty:
                        st.warning(f"No songs found for mood: {mood}")
                    else:
                        cols = st.columns(4) # 4 Columns for cleaner look
                        idx = 0
                        for _, row in recs.iterrows():
                            with cols[idx % 4]:
                                render_song_card(row)
                            idx += 1

# --- PAGE 2: SEARCH & FILTER ---
else:
    st.header("🔍 Discover Music")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name_query = st.text_input("Search Song")
    with col2:
        artist_list = sorted(df['singer'].dropna().unique().tolist())
        artist_filter = st.selectbox("Artist", ["All"] + artist_list)
    with col3:
        mood_list = sorted(df['mood'].dropna().unique().tolist())
        mood_filter = st.selectbox("Mood", ["All"] + mood_list)

    # Apply Filters
    filtered = df.copy()

    if name_query:
        filtered = filtered[filtered['song_name'].str.contains(name_query, case=False, na=False)]
    if artist_filter != "All":
        filtered = filtered[filtered['singer'] == artist_filter]
    if mood_filter != "All":
        filtered = filtered[filtered['mood'] == mood_filter]

    st.markdown("---")
    
    if filtered.empty:
        st.info("No songs found matching your criteria.")
    else:
        st.write(f"Found **{len(filtered)}** songs")
        
        cols = st.columns(4)
        idx = 0
        
        # Limit to 20 to prevent lag
        for _, row in filtered.head(20).iterrows():
            with cols[idx % 4]:
                render_song_card(row)
                
                # CF Button
                if st.button(f"Similar 🎶", key=f"sim_{row.name}"):
                    st.toast(f"Finding songs like {row['song_name']}...")
                    recs = recommend_cf(row['song_name'], 4)
                    if not recs.empty:
                        st.write("---")
                        st.write(f"**Similar to {row['song_name']}:**")
                        s_cols = st.columns(4)
                        s_idx = 0
                        for _, s_row in recs.iterrows():
                            with s_cols[s_idx % 4]:
                                render_song_card(s_row)
                            s_idx += 1
                    else:
                        st.warning("No similar songs data found.")
            idx += 1
