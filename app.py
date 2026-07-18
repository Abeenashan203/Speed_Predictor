import streamlit as st

# 1. Define the Google Drive URLs for both videos
VIDEO_1_ID = "https://drive.google.com/file/d/1L13pk5HG3tgNM1ubrdqe2QGiI-8HPjZB/view?usp=sharing" # The 10.1 MB low-angle car video
VIDEO_2_ID = "https://drive.google.com/file/d/1jWE9FgxGsTSksyQ5YhD4NWn1tZbZYJ1N/view?usp=sharing" # The 8.5 MB blue sedan video

url_1 = f"https://docs.google.com/uc?export=download&id={VIDEO_1_ID}"
url_2 = f"https://docs.google.com/uc?export=download&id={VIDEO_2_ID}"

# 2. HTML/CSS/JS Injection for sequential video playlist looping
playlist_html = f"""
<style>
#bgVideo {{
  position: fixed;
  right: 0;
  bottom: 0;
  min-width: 100%; 
  min-height: 100%;
  width: auto;
  height: auto;
  z-index: -1;
  object-fit: cover;
  opacity: 0.35; /* Keeps text readable */
}}
.main {{
    background: transparent !important;
}}
</style>

<!-- Notice we removed the 'loop' attribute from the HTML so JS can catch the 'ended' event -->
<video id="bgVideo" autoplay muted playsinline>
  <source id="videoSource" src="{url_1}" type="video/mp4">
  Your browser does not support HTML5 video.
</video>

<script>
const video = document.getElementById('bgVideo');
const source = document.getElementById('videoSource');

// Array containing both of your video URLs
const playlist = ["{url_1}", "{url_2}"];
let currentVideoIndex = 0;

// Listen for when the current video finishes playing
video.addEventListener('ended', function() {{
    // Switch to the next video index (loops back to 0 after the last one)
    currentVideoIndex = (currentVideoIndex + 1) % playlist.length;
    
    // Update source and reload player
    source.src = playlist[currentVideoIndex];
    video.load();
    video.play().catch(error => console.log("Playback interrupted:", error));
}});
</script>
"""

# Inject into your Streamlit App
st.markdown(playlist_html, unsafe_allow_html=True)

# 3. Your Vehicle Speed Detector UI
st.title("🚗 VEHICLE SPEED DETECTOR")
st.subheader("Real-time Analytics Dashboard")

# Example dashboard layout
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Average Speed", value="64 mph", delta="2 mph")
with col2:
    st.metric(label="Total Vehicles Detected", value="1,482", delta="120 today")
