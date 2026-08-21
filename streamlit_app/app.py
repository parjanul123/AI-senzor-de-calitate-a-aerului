import sys
from pathlib import Path

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Ensure backend package imports resolve from project root, not from this script folder.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.database import (
    get_device_identifiers,
    get_devices_with_location,
    get_measurements,
    get_supabase_config_status,
    summarize_measurements,
)

API_URL = "http://127.0.0.1:8000/predict"
ANOMALY_API_URL = "http://127.0.0.1:8000/anomaly"
CHAT_API_URL = "http://127.0.0.1:8000/chat"

FALLBACK_SENSOR_DEFAULTS = {
    "temperature": 25.0,
    "humidity": 45.0,
    "pressure": 1013.0,
    "co2": 600.0,
    "pm1": 10.0,
    "pm25": 20.0,
    "pm10": 35.0,
    "voc": 2.0,
    "light": 300.0,
}


def _safe_float(value, fallback):
    try:
        if value is None:
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _get_last_value(row_dict, candidates, fallback):
    for candidate in candidates:
        if candidate in row_dict:
            return _safe_float(row_dict[candidate], fallback)
    return float(fallback)


@st.cache_data(ttl=10)
def get_sensor_defaults_from_database():
    defaults = FALLBACK_SENSOR_DEFAULTS.copy()

    try:
        dataframe = get_measurements(limit=1, descending=True)
        if dataframe.empty:
            return defaults

        latest_row = dataframe.iloc[0].to_dict()
        defaults["temperature"] = _get_last_value(latest_row, ["temperature", "temp"], defaults["temperature"])
        defaults["humidity"] = _get_last_value(latest_row, ["humidity", "umiditate"], defaults["humidity"])
        defaults["pressure"] = _get_last_value(latest_row, ["pressure", "presiune"], defaults["pressure"])
        defaults["co2"] = _get_last_value(latest_row, ["co2", "co_2"], defaults["co2"])
        defaults["pm1"] = _get_last_value(latest_row, ["pm1"], defaults["pm1"])
        defaults["pm25"] = _get_last_value(latest_row, ["pm25", "pm2_5", "pm2.5"], defaults["pm25"])
        defaults["pm10"] = _get_last_value(latest_row, ["pm10"], defaults["pm10"])
        defaults["voc"] = _get_last_value(latest_row, ["voc"], defaults["voc"])
        defaults["light"] = _get_last_value(latest_row, ["light", "lumina"], defaults["light"])
    except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
        return defaults

    return defaults

st.set_page_config(page_title="Air Quality AI", page_icon="🌿", layout="wide")

pages = {
    "Dashboard": "dashboard",
    "Anomaly": "anomaly",
    "Database": "database",
    "Chat": "chat",
    "Settings": "settings",
}

selected_page = st.sidebar.selectbox("Navigation", options=list(pages.keys()))
if selected_page == "Dashboard":
    st.title("Dashboard")
    st.write("Pagina de dashboard este gata pentru integrarea datelor și metricilor.")
elif selected_page == "Database":
    st.title("Database")
    st.write("Încarcă și inspectează datele din tabela measurements (Supabase).")

    with st.expander("Diagnostic conexiune Supabase", expanded=True):
        cfg = get_supabase_config_status()
        st.write(f"Cale fișier .env: {cfg['env_path']}")
        st.write(f".env există: {cfg['env_exists']}")
        st.write(f"SUPABASE_URL setat: {cfg['supabase_url_set']}")
        st.write(f"SUPABASE_SERVICE_ROLE_KEY setat: {cfg['service_role_key_set']}")

        if not cfg["env_exists"]:
            st.warning("Fișierul .env lipsește. Creează-l în rădăcina proiectului.")
        elif not (cfg["supabase_url_set"] and cfg["service_role_key_set"]):
            st.warning("Variabilele din .env sunt incomplete.")

    col1, col2 = st.columns([1, 1])
    with col1:
        auto_refresh = st.toggle("Actualizare în timp real", value=True)
    with col2:
        refresh_seconds = st.slider("Interval refresh (sec)", min_value=2, max_value=60, value=5)

    if auto_refresh:
        st_autorefresh(interval=refresh_seconds * 1000, key="database_autorefresh")

    try:
        device_details = get_devices_with_location()
        if device_details:
            device_ids = [str(item.get("device_identifier", "")).strip() for item in device_details]
            device_ids = [item for item in device_ids if item]
            label_map = {
                str(item.get("device_identifier", "")).strip(): (
                    f"{str(item.get('device_identifier', '')).strip()} - {str(item.get('location_label', '')).strip()}"
                )
                for item in device_details
                if str(item.get("device_identifier", "")).strip()
            }
        else:
            device_ids = get_device_identifiers()
            label_map = {device_id: str(device_id) for device_id in device_ids}

        device_options = ["Toate dispozitivele"] + device_ids if device_ids else ["Toate dispozitivele"]
        selected_device = st.selectbox(
            "Device ID / Nume dispozitiv",
            options=device_options,
            format_func=lambda value: label_map.get(value, value),
        )
        filter_value = None if selected_device == "Toate dispozitivele" else selected_device

        df = get_measurements(device_identifier=filter_value, limit=20, descending=True, raise_on_error=True)
        summary = summarize_measurements(df)

        st.info(f"Număr total de înregistrări afișate: {summary['row_count']}")
        st.write("Coloane detectate în measurements:")
        st.write(summary["columns"])

        if summary["row_count"] == 0:
            st.warning(
                "Nu există date pentru filtrul selectat sau conexiunea la Supabase nu este configurată corect."
            )
        else:
            if "created_at" in df.columns and df["created_at"].notna().any():
                st.caption(f"Ultima actualizare: {df['created_at'].max()}")

            st.subheader("Primele 20 de înregistrări, ordonate cu cele mai noi primele")
            st.dataframe(df, use_container_width=True, height=700)

            st.subheader("Informații despre coloane și tipuri de date")
            dtypes_df = (
                df.dtypes.astype(str)
                .rename("dtype")
                .reset_index()
                .rename(columns={"index": "column"})
            )
            st.table(dtypes_df)
    except RuntimeError as exc:
        st.error(f"Eroare la încărcarea datelor din baza de date: {exc}")
        st.code(str(exc), language="text")
elif selected_page == "Chat":
    st.title("Chat")
    st.write("Asistent rule-based pentru întrebări despre calitatea aerului.")

    with st.form("chat_form"):
        user_message = st.text_area(
            "Mesaj",
            placeholder="Ex: Cum interpretez valorile PM2.5 și CO2?",
            height=120,
        )
        submitted = st.form_submit_button("Trimite")

    if submitted:
        try:
            response = requests.post(
                CHAT_API_URL,
                json={"message": user_message},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            st.success("Răspuns primit")
            assistant_reply = result.get("text") or result.get("reply") or "Nu am putut obține un răspuns."
            st.write(f"Asistent: {assistant_reply}")
            if result.get("selected"):
                st.caption(f"Dispozitiv selectat: {result.get('selected')}")
        except requests.RequestException as exc:
            st.error(f"Eroare la comunicarea cu endpointul /chat: {exc}")
else:
    st.title("Settings")
    st.write("Setările aplicației vor fi configurate aici.")
