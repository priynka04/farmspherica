import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ── Page config (must be first Streamlit command) ──────────────────────────
st.set_page_config(
    page_title="Farmspherica Dashboard",
    page_icon="🌱",
    layout="wide"
)

API_URL       = "http://localhost:8001"   # dashboard API
IMAGE_API_URL = "http://localhost:8002"   # image API
RAG_API_URL   = "http://localhost:8000"   # RAG API

st.title("🌱 Farmspherica Nano PAW — Live Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y %H:%M:%S')}")

if st.button("🔄 Refresh data"):
    st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# SECTION 1: ALERTS
# ────────────────────────────────────────────────────────────────────────────
st.subheader("⚠️ Alerts")
try:
    alerts_resp = requests.get(f"{API_URL}/alerts", timeout=5)
    alerts_data = alerts_resp.json()
    if alerts_data["count"] == 0:
        st.success("✅ All sensors are within safe range.")
    else:
        for alert in alerts_data["alerts"]:
            st.error(f"🚨 {alert['message']}")
except Exception as e:
    st.warning(f"Could not reach API: {e}. Make sure your dashboard_api.py is running.")

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# SECTION 2: LATEST SENSOR READINGS (Cards)
# ────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Latest Sensor Readings")
try:
    latest_resp = requests.get(f"{API_URL}/data/latest", timeout=5)
    latest = latest_resp.json()

    col1, col2, col3, col4, col5 = st.columns(5)

    def show_card(col, label, key, unit, safe_low, safe_high):
        val = latest.get(key, "N/A")
        with col:
            if val != "N/A" and val is not None:
                val = round(float(val), 2)
                status = "🟢" if safe_low <= val <= safe_high else "🔴"
                st.metric(label=f"{status} {label}", value=f"{val} {unit}")
            else:
                st.metric(label=label, value="No data")

    show_card(col1, "pH",        "pH",              "",       4.0, 9.0)
    show_card(col2, "EC",        "EC",              "mS/cm",  0.0, 5.0)
    show_card(col3, "Water Temp","water_temp_C",    "°C",     10,  35)
    show_card(col4, "Height",    "plant_height_cm", "cm",     0,   300)
    show_card(col5, "Leaf Count","leaf_count",      "leaves", 0,   500)

except Exception as e:
    st.warning(f"Could not load latest readings: {e}")

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# SECTION 3: 7-DAY TREND CHARTS 
# ────────────────────────────────────────────────────────────────────────────
st.subheader("📈 7-Day Trends")
try:
    trends_resp = requests.get(f"{API_URL}/data/trends", timeout=5)
    trends = trends_resp.json()
    df = pd.DataFrame(trends)

    if not df.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            if "pH" in df.columns:
                fig = px.line(df, x=df.index, y="pH", title="pH over time",
                              markers=True, color_discrete_sequence=["#1a6b3c"])
                fig.add_hline(y=4.0, line_dash="dot", line_color="red",
                              annotation_text="Min safe")
                fig.add_hline(y=9.0, line_dash="dot", line_color="red",
                              annotation_text="Max safe")
                st.plotly_chart(fig, use_container_width=True)

            if "plant_height_cm" in df.columns:
                fig2 = px.bar(df, x=df.index, y="plant_height_cm",
                              title="Plant height (cm)",
                              color_discrete_sequence=["#2e7d32"])
                st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            if "EC" in df.columns:
                fig3 = px.line(df, x=df.index, y="EC", title="EC over time",
                               markers=True, color_discrete_sequence=["#1565c0"])
                fig3.add_hline(y=0.0, line_dash="dot", line_color="red",
                               annotation_text="Min safe")
                fig3.add_hline(y=5.0, line_dash="dot", line_color="red",
                               annotation_text="Max safe")
                st.plotly_chart(fig3, use_container_width=True)

            if "water_temp_C" in df.columns:
                fig4 = px.line(df, x=df.index, y="water_temp_C",
                               title="Water temperature (°C)",
                               markers=True, color_discrete_sequence=["#e65100"])
                st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No trend data yet.")

except Exception as e:
    st.warning(f"Could not load trend data: {e}")

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# SECTION 4: PLANT GROWTH TABLE
# ────────────────────────────────────────────────────────────────────────────
st.subheader("🌿 All Plant Records")
try:
    plants_resp = requests.get(f"{API_URL}/plants", timeout=5)
    plants      = plants_resp.json()
    df_plants   = pd.DataFrame(plants["records"])
    if not df_plants.empty:
        cols_to_show = [c for c in
            ["date", "day_number", "plant_id", "pH", "EC",
             "water_temp_C", "plant_height_cm", "leaf_count",
             "condition", "remarks"]
            if c in df_plants.columns]
        st.dataframe(df_plants[cols_to_show], use_container_width=True)
    else:
        st.info("No plant records found.")
except Exception as e:
    st.warning(f"Could not load plant records: {e}")

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# SECTION 5: IMAGE LOGGING SYSTEM
# ────────────────────────────────────────────────────────────────────────────
st.subheader("📷 Plant Photo Logger")

tab_upload, tab_gallery = st.tabs(["Upload a Photo", "Photo Gallery"])

with tab_upload:
    st.write("Upload a new plant photo here.")
    uploaded_file = st.file_uploader(
        "Choose a photo (jpg, png)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            plant_id  = st.text_input("Plant ID", value="P01")
        with col_b:
            condition = st.selectbox("Condition",
                ["Healthy", "Mildly Stressed", "Deficient", "Critical"])
        with col_c:
            angle = st.selectbox("Angle", ["Front", "Side", "Root"])
        photo_date = st.date_input("Date of photo")
        notes      = st.text_input("Notes (optional)", value="")

        if st.button("📤 Upload Photo"):
            try:
                resp = requests.post(
                    f"{IMAGE_API_URL}/photos/upload",
                    files={"file": (uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "image/jpeg")},
                    data={
                        "plant_id":  plant_id,
                        "condition": condition,
                        "angle":     angle,
                        "notes":     notes,
                        "date":      str(photo_date)
                    },
                    timeout=30
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(f"✅ Photo uploaded: {result['filename']}")
                    # Show CV model prediction result
                    cv = result.get("cv_prediction", {})
                    if cv.get("predicted_class"):
                        st.info(
                            f"🤖 CV Model predicted: **{cv['predicted_class']}** "
                            f"({cv.get('confidence_pct', '')} confidence)"
                        )
                        if cv.get("all_probabilities"):
                            st.json(cv["all_probabilities"])
                else:
                    st.error(f"Upload failed: {resp.text}")
            except Exception as e:
                st.warning(
                    f"Image API not running: {e}. "
                    f"Start it with: uvicorn api.image_api:app --port 8002"
                )

with tab_gallery:
    st.write("Browse all uploaded plant photos.")
    filter_condition = st.selectbox(
        "Filter by condition",
        ["All", "Healthy", "Mildly Stressed", "Deficient", "Critical"],
        key="gallery_filter"
    )
    try:
        params = {}
        if filter_condition != "All":
            params["condition"] = filter_condition

        gallery_resp = requests.get(
            f"{IMAGE_API_URL}/photos/list",
            params=params,
            timeout=5
        )
        data   = gallery_resp.json()
        photos = data.get("photos", [])

        if not photos:
            st.info("No photos uploaded yet.")
        else:
            st.write(f"Showing {len(photos)} photo(s)")
            cols = st.columns(3)
            for i, photo in enumerate(photos):
                with cols[i % 3]:
                    # Fetch image using the correct endpoint
                    img_url = f"{IMAGE_API_URL}/photos/{photo['id']}/image"
                    try:
                        img_resp = requests.get(img_url, timeout=5)
                        if img_resp.status_code == 200:
                            st.image(img_resp.content, width=200)
                        else:
                            st.write("📷 (image not found)")
                    except Exception:
                        st.write("📷 (could not load image)")

                    # Manual label
                    st.caption(f"**{photo['condition']}** | {photo['date']}")
                    st.caption(f"Plant: {photo['plant_id']} | Angle: {photo['angle']}")

                    # CV model prediction (Week 4 addition)
                    if photo.get("predicted_condition"):
                        conf = photo.get("prediction_confidence") or 0
                        st.caption(
                            f"🤖 CV Model: **{photo['predicted_condition']}** "
                            f"({conf * 100:.1f}%)"
                        )

                    if photo.get("notes"):
                        st.caption(f"Notes: {photo['notes']}")

    except Exception as e:
        st.warning(f"Could not reach Image API: {e}")

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# SECTION 6: RAG SMART FARMING ASSISTANT
# ────────────────────────────────────────────────────────────────────────────
st.subheader("🤖 Ask the Smart Farming Assistant")
st.caption("Ask any question about hydroponics, plant health, nutrients, or your sensor data.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources cited"):
                for src in msg["sources"]:
                    st.write(f"• {src}")

user_question = st.chat_input("Ask a farming question...")

if user_question:
    st.chat_message("user").write(user_question)
    st.session_state.chat_history.append({
        "role":    "user",
        "content": user_question
    })

    try:
        with st.spinner("Thinking..."):
            rag_resp = requests.post(
                f"{RAG_API_URL}/ask",
                json={"question": user_question},
                timeout=30
            )
        if rag_resp.status_code == 200:
            result  = rag_resp.json()
            answer  = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])

            st.chat_message("assistant").write(answer)
            if sources:
                with st.expander("📚 Sources cited"):
                    for src in sources:
                        st.write(f"• {src}")

            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": answer,
                "sources": sources
            })
        else:
            st.error(f"RAG API error: {rag_resp.text}")
    except Exception as e:
        st.warning(
            f"Could not reach RAG API: {e}. "
            f"Start it with: uvicorn api.rag_api:app --reload --port 8000"
        )

if st.button("🗑️ Clear chat history"):
    st.session_state.chat_history = []
    try:
        requests.post(f"{RAG_API_URL}/clear-memory", timeout=5)
    except Exception:
        pass
    st.rerun()
    
st.divider()

# ── GROWTH FORECAST PANEL ──────────────────────────────────────────────────
st.subheader("🌿 Growth Forecast")
st.caption("Enter current sensor readings to predict plant height.")

import joblib as jl
import os

if not os.path.exists("models/growth_model.pkl"):
    st.info("Growth model not trained yet. Run: python api/growth_model.py")
else:
    growth_model    = jl.load("models/growth_model.pkl")
    growth_features = jl.load("models/growth_model_features.pkl")

    col1, col2, col3 = st.columns(3)
    with col1:
        g_day        = st.number_input("Day after transplant", 0,  70,  7)
        g_ph         = st.number_input("pH",                   4.0, 9.0, 6.0, step=0.1)
        g_ec         = st.number_input("EC (mS/cm)",           0.0, 5.0, 1.2, step=0.1)
        g_tds        = st.number_input("TDS (ppm)",            0,   3000, 800)
    with col2:
        g_wtemp      = st.number_input("Water temp (°C)",      10.0, 35.0, 20.0, step=0.5)
        g_atemp      = st.number_input("Ambient temp (°C)",    10.0, 40.0, 21.0, step=0.5)
        g_do         = st.number_input("DO (mg/L)",            0.0,  15.0, 7.0,  step=0.1)
        g_humidity   = st.number_input("Humidity (%)",         0,    100,  65)
    with col3:
        g_photo      = st.number_input("Photoperiod (hrs)",    0.0,  24.0, 14.0, step=0.5)
        g_ppfd       = st.number_input("PPFD (umol)",          0,    500,  220)
        g_leaves     = st.number_input("Leaf count",           1,    50,   8)
        g_stage      = st.selectbox("Growth stage", ["establishment", "vegetative", "generative", "harvest"])
        g_crop       = st.selectbox("Crop type", ["lettuce", "strawberry"])

    if st.button("Predict Height"):
        from sklearn.preprocessing import LabelEncoder
        le = jl.load("models/growth_stage_encoder.pkl")
        try:
            stage_enc = le.transform([g_stage])[0]
        except Exception:
            stage_enc = 1
        crop_enc = 1 if g_crop == "strawberry" else 0

        row  = [[g_day, g_wtemp, g_atemp, g_ph, g_ec, g_tds,
                 g_do, g_humidity, g_photo, g_ppfd, g_leaves,
                 stage_enc, crop_enc]]
        pred = growth_model.predict(row)[0]
        st.success(f"Predicted plant height: **{pred:.1f} cm**")
        st.caption("Based on Random Forest / XGBoost trained on 1785 rows of lettuce + strawberry data")