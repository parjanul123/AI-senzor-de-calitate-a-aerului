import sys
import traceback
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
SCRIPT_DIR_PATH = Path(__file__).resolve().parent
cleaned_sys_path = []
for entry in sys.path:
    try:
        resolved_entry = Path(entry).resolve() if entry else Path.cwd().resolve()
    except (OSError, RuntimeError, ValueError):
        cleaned_sys_path.append(entry)
        continue
    if resolved_entry != SCRIPT_DIR_PATH:
        cleaned_sys_path.append(entry)
sys.path = cleaned_sys_path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

existing_app_module = sys.modules.get("app")
existing_app_file = getattr(existing_app_module, "__file__", "") if existing_app_module else ""
if existing_app_file and str(SCRIPT_DIR_PATH) in str(existing_app_file):
    del sys.modules["app"]

from app.core.database import (
    get_device_identifiers,
    get_measurements,
    get_supabase_config_status,
    summarize_measurements,
)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{BACKEND_BASE_URL}/predict"
API_DEMO_URL = f"{BACKEND_BASE_URL}/predict-demo"
API_CUSTOM_URL = f"{BACKEND_BASE_URL}/predict-custom"
TRAIN_API_URL = f"{BACKEND_BASE_URL}/train"
TRAIN_DEMO_API_URL = f"{BACKEND_BASE_URL}/train-demo"
ANOMALY_API_URL = f"{BACKEND_BASE_URL}/anomaly"
CHAT_API_URL = f"{BACKEND_BASE_URL}/chat"

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
    "Predict": "predict",
    "Train": "train",
    "Anomaly": "anomaly",
    "Database": "database",
    "Chat": "chat",
    "Settings": "settings",
}

selected_page = st.sidebar.selectbox("Navigation", options=list(pages.keys()))

if selected_page == "Dashboard":
    st.title("Dashboard")
    st.write("Pagina de dashboard este gata pentru integrarea datelor și metricilor.")
elif selected_page == "Predict":
    st.title("Predict")
    
    # Choose prediction mode
    prediction_mode = st.radio(
        "Selectează modul de predicție:",
        options=["Real (Supabase)", "Demo (test)", "Custom (manual)"],
        horizontal=True
    )
    
    if prediction_mode == "Real (Supabase)":
        st.write("Predicția folosește ultima înregistrare din tabela measurements din Supabase.")
        use_hourly_average = st.toggle("Folosește medie pe interval orar", value=False)
        aggregation_hours = st.slider("Interval orar pentru predicție (ore)", min_value=1, max_value=168, value=1)
        future_hours = st.slider("Ore viitoare pentru prognoză", min_value=0, max_value=48, value=0)

        if st.button("Generează predicție"):
            try:
                request_url = API_URL
                query_parts = []
                if use_hourly_average:
                    query_parts.append("use_hourly_average=true")
                    query_parts.append(f"aggregation_hours={aggregation_hours}")
                if future_hours > 0:
                    query_parts.append("include_forecast=true")
                    query_parts.append(f"forecast_horizons={future_hours}")

                if query_parts:
                    request_url = f"{API_URL}?{'&'.join(query_parts)}"

                response = requests.post(request_url, timeout=10)
                response.raise_for_status()
                result = response.json()

                st.success("Predicție primită")
                st.write(f"Status: {result.get('status')}")
                st.write(f"Mesaj: {result.get('message')}")
                st.write(f"Predicție: {result.get('prediction')}")
                st.write(f"Încredere: {result.get('confidence')}")

                st.subheader("Valorile folosite pentru predicție")
                st.json(result.get("input_values", {}))

                feature_assessment = result.get("feature_assessment") or {}
                if feature_assessment:
                    st.subheader("Evaluare pe fiecare parametru")
                    assessment_rows = []
                    for feature_name, details in feature_assessment.items():
                        assessment_rows.append(
                            {
                                "parametru": feature_name,
                                "valoare": details.get("value"),
                                "unitate": details.get("unit"),
                                "status": details.get("status"),
                                "human_condition": details.get("human_condition"),
                            }
                        )
                    if assessment_rows:
                        st.dataframe(pd.DataFrame(assessment_rows))

            except requests.exceptions.JSONDecodeError:
                st.error("Eroare: Răspuns invalid de la API. Verifică dacă serverul rulează și dacă baza de date are date.")
            except requests.RequestException as exc:
                st.error(f"Eroare la comunicarea cu API-ul /predict: {exc}")
                
    elif prediction_mode == "Demo (test)":
        st.write("Predicție cu date de test - util pentru testing.")
        st.info("Folosește date pre-definite: T=22.5°C, H=55%, PM2.5=18.5, PM10=35.2, CO2=950ppm")
        
        if st.button("Generează predicție DEMO"):
            try:
                response = requests.post(API_DEMO_URL, timeout=10)
                response.raise_for_status()
                result = response.json()

                st.success("Predicție DEMO primită")
                st.write(f"Predicție: {result.get('prediction')}")
                st.write(f"Încredere: {result.get('confidence')}")
                st.json(result.get("input_values", {}))
                
            except requests.RequestException as exc:
                st.error(f"Eroare la predicția demo: {exc}")
                
    else:  # Custom
        st.write("Introduceți manual valorile pentru predicție:")
        
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input("Temperatura (°C)", value=22.5, min_value=-50.0, max_value=60.0)
            humidity = st.number_input("Umiditate (%)", value=55.0, min_value=0.0, max_value=100.0)
            pm25 = st.number_input("PM2.5 (µg/m³)", value=18.5, min_value=0.0, max_value=500.0)
        with col2:
            pm10 = st.number_input("PM10 (µg/m³)", value=35.2, min_value=0.0, max_value=500.0)
            co2 = st.number_input("CO2 (ppm)", value=950.0, min_value=400.0, max_value=5000.0)
        
        if st.button("Generează predicție CUSTOM"):
            try:
                payload = {
                    "temperature": temperature,
                    "humidity": humidity,
                    "pm25": pm25,
                    "pm10": pm10,
                    "co2": co2
                }
                response = requests.post(API_CUSTOM_URL, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()

                st.success("Predicție CUSTOM primită")
                st.write(f"Predicție: {result.get('prediction')}")
                st.write(f"Încredere: {result.get('confidence')}")
                st.json(result.get("input_values", {}))
                
            except requests.RequestException as exc:
                st.error(f"Eroare la predicția custom: {exc}")

elif selected_page == "Train":
    st.title("Train")
    st.write("Antrenarea poate folosi interval pe ore sau pe minute pentru o granularitate mai fină.")

    def _fmt_ts(value):
        if not value:
            return "N/A"
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return str(value)
        return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Training mode selector
    training_mode = st.radio(
        "Selectează modul de antrenare:",
        options=["Real (Supabase)", "Demo (test)"],
        horizontal=True
    )

    if training_mode == "Real (Supabase)":
        st.subheader("Selectare algoritm")
        selected_model = st.radio(
            "Alege algoritmul pentru antrenare:",
            options=["random_forest", "xgboost", "svm", "isolation_forest"],
            format_func=lambda x: {
                "random_forest": "🌳 Random Forest",
                "xgboost": "🚀 XGBoost",
                "svm": "🎯 SVM",
                "isolation_forest": "🔍 Isolation Forest",
            }[x],
        )
        training_granularity = st.selectbox(
            "Granularitate antrenare",
            options=["ore", "minute"],
            index=0,
        )

        training_aggregation_hours = 24
        training_aggregation_minutes = None
        if training_granularity == "minute":
            training_aggregation_minutes = st.slider(
                "Interval pentru antrenare (minute)",
                min_value=5,
                max_value=60,
                value=30,
                step=5,
            )
        else:
            training_aggregation_hours = st.slider(
                "Interval pentru antrenare (ore)",
                min_value=1,
                max_value=720,
                value=24,
            )

        if st.button("Antrenează modelul", type="primary"):
            try:
                response = requests.post(
                    TRAIN_API_URL,
                    json={
                        "training_model": selected_model,
                        "aggregation_hours": training_aggregation_hours,
                        "aggregation_minutes": training_aggregation_minutes,
                    },
                    timeout=90,
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.latest_training_result = result

                st.success("Antrenare finalizată")
                st.write(result.get("message", ""))
                
            except requests.RequestException as exc:
                st.error(f"Eroare la antrenare: {exc}")
    else:  # Demo mode
        st.info("Antrenare DEMO cu Random Forest pe date din Supabase (24h agregare)")
        
        if st.button("Antrenează modelul DEMO", type="primary"):
            try:
                response = requests.post(TRAIN_DEMO_API_URL, timeout=90)
                response.raise_for_status()
                result = response.json()
                st.session_state.latest_training_result = result

                st.success("Antrenare DEMO finalizată")
                st.write(result.get("message", ""))
                
                training_report = result.get("training_report", {})
                dataset_info = training_report.get("dataset_info", {})
                model_info = training_report.get("model_info", {})
                evaluation = training_report.get("evaluation")
                evaluation_note = training_report.get("evaluation_note")
                anomaly_summary = training_report.get("anomaly_summary", {})
                technical_details = training_report.get("technical_details", {})
                summary = training_report.get("summary", {})

                st.subheader("1) Setul de date utilizat")
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.metric("Dispozitive sursă", int(dataset_info.get("device_count", 0)))
                with d_col2:
                    agg_granularity = summary.get("aggregation_granularity")
                    agg_value = summary.get("aggregation_value")
                    if agg_granularity == "minute" and agg_value is not None:
                        st.metric("Agregare", f"{int(agg_value)} minute")
                    else:
                        st.metric("Agregare", f"{int(summary.get('aggregation_hours', training_aggregation_hours))} ore")

                time_range = dataset_info.get("time_range", {})
                st.write(
                    "Interval temporal: "
                    f"{_fmt_ts(time_range.get('start'))}  ->  {_fmt_ts(time_range.get('end'))}"
                )

                class_distribution = dataset_info.get("class_distribution")
                if class_distribution:
                    st.markdown("**Distribuția claselor (good / moderate / poor)**")
                    class_df = pd.DataFrame(
                        [{"class": key, "count": int(value)} for key, value in class_distribution.items()]
                    )
                    st.dataframe(class_df, use_container_width=True)

                st.subheader("2) Informații despre model")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("Model", model_info.get("name", result.get("model_type", "N/A")))
                with m_col2:
                    n_estimators = model_info.get("n_estimators")
                    st.metric("Număr arbori", n_estimators if n_estimators is not None else "N/A")
                with m_col3:
                    st.metric("Ultima antrenare", _fmt_ts(model_info.get("last_trained_at")))

                model_path = model_info.get("model_path")
                if model_path:
                    st.write(f"Fișier model (.pkl): {model_path}")

                st.subheader("3) Evaluarea modelului")
                label_source = summary.get("label_source")
                model_type = result.get("model_type")
                
                # Display evaluation metrics for all supervised models
                if model_type in ["random_forest", "xgboost", "svm"]:
                    if label_source == "database_quality_label" and evaluation:
                        # Performance metrics
                        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
                        with e_col1:
                            st.metric("Accuracy", f"{evaluation.get('accuracy', 0.0):.4f}")
                        with e_col2:
                            st.metric("Precision", f"{evaluation.get('precision', 0.0):.4f}")
                        with e_col3:
                            st.metric("Recall", f"{evaluation.get('recall', 0.0):.4f}")
                        with e_col4:
                            st.metric("F1-score", f"{evaluation.get('f1_score', 0.0):.4f}")

                        # Confusion Matrix
                        confusion_matrix_data = evaluation.get("confusion_matrix", {})
                        labels_list = confusion_matrix_data.get("labels", [])
                        matrix = confusion_matrix_data.get("matrix", [])
                        if labels_list and matrix:
                            st.markdown("**Confusion Matrix**")
                            cm_df = pd.DataFrame(matrix, index=labels_list, columns=labels_list)
                            st.dataframe(cm_df, use_container_width=True)
                        
                        # Classification Report
                        classification_report_data = evaluation.get("classification_report", {})
                        if classification_report_data:
                            with st.expander("Classification Report (Detaliat)", expanded=False):
                                report_rows = []
                                for class_label, metrics in classification_report_data.items():
                                    if isinstance(metrics, dict) and "precision" in metrics:
                                        report_rows.append({
                                            "class": class_label,
                                            "precision": f"{metrics.get('precision', 0.0):.4f}",
                                            "recall": f"{metrics.get('recall', 0.0):.4f}",
                                            "f1-score": f"{metrics.get('f1-score', 0.0):.4f}",
                                            "support": int(metrics.get('support', 0)),
                                        })
                                if report_rows:
                                    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
                    else:
                        explanation = evaluation_note or (
                            "Metricile clasice nu sunt disponibile pentru această sesiune de antrenare."
                        )
                        st.info(explanation)

                # Feature Importance (for models that support it)
                if model_type in ["random_forest", "xgboost"]:
                    feature_importances = technical_details.get("feature_importances", {})
                    if feature_importances:
                        st.markdown("**Feature Importance**")
                        importance_df = pd.DataFrame([
                            {"Feature": k, "Importance": float(v)}
                            for k, v in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.bar_chart(importance_df.set_index("Feature"))
                        with st.expander("Valori Feature Importance", expanded=False):
                            st.dataframe(importance_df, use_container_width=True)

                if result.get("model_type") == "isolation_forest":
                    st.subheader("4) Rezultate Isolation Forest")
                    a_col1, a_col2, a_col3 = st.columns(3)
                    with a_col1:
                        st.metric("Anomalii detectate", int(anomaly_summary.get("anomaly_count", 0)))
                    with a_col2:
                        st.metric("Procent anomalii", f"{anomaly_summary.get('anomaly_percentage', 0.0):.2f}%")
                    with a_col3:
                        st.metric("Contamination", anomaly_summary.get("contamination", "N/A"))

                    distribution = anomaly_summary.get("distribution", {})
                    if distribution:
                        dist_df = pd.DataFrame(
                            {
                                "category": ["normal", "anomaly"],
                                "count": [
                                    int(distribution.get("normal", 0)),
                                    int(distribution.get("anomaly", 0)),
                                ],
                            }
                        )
                        st.markdown("**Distribuția anomaliilor**")
                        st.bar_chart(dist_df.set_index("category"))

                with st.expander("Detalii tehnice", expanded=False):
                    evolution = technical_details.get("evolution", [])
                    if evolution:
                        evolution_df = pd.DataFrame(evolution)
                        st.dataframe(evolution_df, use_container_width=True)
                        if "oob_score" in evolution_df.columns:
                            oob_series = pd.to_numeric(evolution_df["oob_score"], errors="coerce")
                            if oob_series.notna().any():
                                st.line_chart(pd.DataFrame({"oob_score": oob_series}).set_index(evolution_df["step"]))
                            else:
                                rows_used = int(summary.get("rows_used", 0) or 0)
                                st.info(
                                    "OOB score nu este disponibil pentru acest training. "
                                    f"Setul curent are {rows_used} rânduri agregate; recomand cel puțin 10 pentru o estimare stabilă."
                                )
                        elif "mean_decision_score" in evolution_df.columns:
                            st.line_chart(evolution_df.set_index("step")["mean_decision_score"])
                    else:
                        st.caption("Nu există detalii tehnice suplimentare pentru această antrenare.")
                
            except requests.RequestException as exc:
                st.error(f"Eroare la antrenare demo: {exc}")

    latest_training_result = st.session_state.get("latest_training_result")
    if latest_training_result:
        latest_training_report = latest_training_result.get("training_report", {})
        latest_model_info = latest_training_report.get("model_info", {})
        latest_evolution = latest_training_report.get("technical_details", {}).get("evolution", [])
        if latest_evolution:
            st.subheader("Evoluția antrenării")
            iteration_col, estimator_col = st.columns(2)
            with iteration_col:
                st.metric("Număr iterații", len(latest_evolution))
            with estimator_col:
                st.metric("Număr arbori", latest_model_info.get("n_estimators", "N/A"))

            latest_evolution_df = pd.DataFrame(latest_evolution)
            if "oob_score" in latest_evolution_df.columns:
                latest_oob_scores = pd.to_numeric(latest_evolution_df["oob_score"], errors="coerce")
                if latest_oob_scores.notna().any():
                    st.line_chart(
                        pd.DataFrame({"OOB score": latest_oob_scores}).set_index(latest_evolution_df["step"])
                    )
            elif "mean_decision_score" in latest_evolution_df.columns:
                st.line_chart(latest_evolution_df.set_index("step")["mean_decision_score"])

        with st.expander("Ultimul raport de antrenare", expanded=training_mode == "Real (Supabase)"):
            st.json(latest_training_report or latest_training_result)
elif selected_page == "Anomaly":
    st.title("Anomaly")
    st.write("Detecția folosește ultima înregistrare din tabela measurements din Supabase.")

    if st.button("Detectează anomalia"):
        try:
            response = requests.post(ANOMALY_API_URL, timeout=10)
            response.raise_for_status()
            result = response.json()
            anomaly_data = result.get("result", {})

            st.success("Rezultat detecție primit")
            st.write(f"Status: {result.get('status')}")
            st.write(f"Mesaj: {result.get('message')}")
            st.write(f"Etichetă: {anomaly_data.get('label')}")
            st.write(f"Este anomalie: {anomaly_data.get('is_anomaly')}")
            st.write(f"Scor anomalie: {anomaly_data.get('score')}")

            st.subheader("Valorile folosite pentru anomaly")
            st.json(anomaly_data.get("input_values", {}))

            suspicious = anomaly_data.get("anomalous_features", [])
            if suspicious:
                st.warning(f"Feature-uri suspecte: {', '.join(suspicious)}")
            else:
                st.info("Nu au fost identificate feature-uri individuale ieșite din intervalul normal.")

            sensor_health_warnings = anomaly_data.get("sensor_health_warnings", [])
            for warning_message in sensor_health_warnings:
                st.warning(warning_message)

            analysis_rows = anomaly_data.get("feature_analysis", [])
            if analysis_rows:
                st.subheader("Analiză pe fiecare feature")
                st.dataframe(analysis_rows, use_container_width=True)
        except requests.RequestException as exc:
            st.error(f"Eroare la comunicarea cu API-ul: {exc}")
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
        devices = get_device_identifiers()
        device_options = ["Toate dispozitivele"] + devices if devices else ["Toate dispozitivele"]
        selected_device = st.selectbox("Device ID / Nume dispozitiv", options=device_options)
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
    except Exception as exc:
        st.error(f"Eroare la încărcarea datelor din baza de date: {exc}")
        st.code(traceback.format_exc(), language="text")
elif selected_page == "Chat":
    st.title("Chat")
    st.write("Asistent rule-based pentru întrebări despre calitatea aerului.")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

    # Input area with columns for better layout
    input_col, button_col = st.columns([5, 1])
    with input_col:
        user_input = st.text_input(
            "Mesaj",
            placeholder="Ex: Cum interpretez valorile PM2.5 și CO2?",
            key="chat_input"
        )
    with button_col:
        send_button = st.button("Trimite", key="send_btn", use_container_width=True)

    # Process message when send button is clicked
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Get response from API
        try:
            response = requests.post(
                CHAT_API_URL,
                json={
                    "message": user_input,
                    "history": st.session_state.chat_history[:-1][-12:],
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            assistant_reply = result.get('reply', 'Nu am putut obține un răspuns.')

            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_reply
            })

            # Clear input by rerunning
            st.rerun()
        except requests.RequestException as exc:
            error_msg = f"Eroare la comunicarea cu asistentul: {exc}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg
            })
            st.rerun()
        except Exception as exc:
            st.error(f"Eroare neașteptată: {exc}")
else:
    st.title("Settings")
    st.subheader("Status Railway")
    st.caption("Verificare live a serviciului backend configurat pentru această interfață.")

    if st.button("Actualizează statusul Railway", type="primary"):
        st.cache_data.clear()

    started_at = time.perf_counter()
    try:
        backend_response = requests.get(f"{BACKEND_BASE_URL}/health", timeout=10)
        response_time_ms = (time.perf_counter() - started_at) * 1000
        backend_status = backend_response.json()
        is_healthy = backend_response.ok and backend_status.get("status") == "ok"
    except (requests.RequestException, ValueError) as exc:
        backend_response = None
        response_time_ms = None
        backend_status = {"error": str(exc)}
        is_healthy = False

    railway_col, status_col, latency_col = st.columns(3)
    with railway_col:
        st.metric("Backend configurat", "Railway" if ".railway.app" in BACKEND_BASE_URL else "Alt server")
    with status_col:
        st.metric("Status API", "Online" if is_healthy else "Indisponibil")
    with latency_col:
        st.metric("Timp răspuns", f"{response_time_ms:.0f} ms" if response_time_ms is not None else "N/A")

    st.code(BACKEND_BASE_URL, language=None)
    if is_healthy:
        st.success(f"Railway răspunde: {backend_status.get('service', 'serviciu disponibil')}")
    else:
        st.error(f"Railway nu răspunde corect: {backend_status.get('error', backend_status)}")

    st.link_button("Deschide Railway Dashboard", "https://railway.app/dashboard")
    st.caption("CPU, RAM, trafic și istoricul deploy-urilor se văd în dashboard-ul Railway pentru serviciul selectat.")

    st.subheader("Configurație aplicație")
    cfg = get_supabase_config_status()
    st.write(f".env: {cfg['env_path']}")
    st.write(f".env există: {cfg['env_exists']}")
    st.write(f"SUPABASE_URL setat: {cfg['supabase_url_set']}")
    st.write(f"SUPABASE_SERVICE_ROLE_KEY setat: {cfg['service_role_key_set']}")
