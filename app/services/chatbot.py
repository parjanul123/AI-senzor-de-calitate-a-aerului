from dataclasses import dataclass
from contextlib import suppress
import json
import re
from typing import Any

import pandas as pd
import requests

from app.core.config import (
    CHATBOT_USE_OLLAMA,
    CHATBOT_ENABLE_LEARNING,
    CHATBOT_ENABLE_WEB_SEARCH,
    CHATBOT_WEB_SEARCH_MAX_SNIPPETS,
    CHATBOT_WEB_SEARCH_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_REPEAT_PENALTY,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_TOP_P,
)
from app.core.database import (
    get_chatbot_notes,
    get_device_location_details,
    get_devices_with_location,
    get_measurements,
    save_chatbot_note,
)
from app.services.bert_sensor_detector import detect_sensor_features, get_bert_model_name
from app.services.predictor import build_forecast, predict_air_quality


WELCOME_MESSAGE = "Bună, sunt AeroSense, agentul tău pentru temperatura și calitatea aerului."


AIR_QUALITY_REFERENCE: dict[str, dict[str, str]] = {
    "temperature": {
        "name": "T (temperatura aerului)",
        "unit": "°C",
        "low": "<18",
        "normal": "20-26",
        "high": ">28",
        "meaning": "arată cât de cald sau rece este aerul din încăpere",
        "effects": "temperaturile prea mici sau prea mari pot da disconfort, oboseală și scăderea concentrării",
        "advice_low": "crește temperatura treptat spre zona de confort (aprox. 20-24°C)",
        "advice_high": "aerisește, redu sursele de căldură sau pornește răcirea",
    },
    "humidity": {
        "name": "H (umiditatea relativă)",
        "unit": "%",
        "low": "<30",
        "normal": "40-60",
        "high": ">70",
        "meaning": "arată câtă umiditate există în aer raportat la maximul posibil",
        "effects": "umiditatea mică usucă ochii/gâtul, iar umiditatea mare favorizează mucegaiul și senzația de aer greu",
        "advice_low": "folosește un umidificator sau surse blânde de umiditate",
        "advice_high": "aerisește des și folosește dezumidificator dacă e nevoie",
    },
    "pressure": {
        "name": "P (presiunea atmosferică)",
        "unit": "hPa",
        "low": "<980",
        "normal": "980-1030",
        "high": ">1030",
        "meaning": "reflectă presiunea aerului din atmosferă și poate indica schimbări de vreme",
        "effects": "variațiile bruște pot produce disconfort la persoane sensibile (cefalee, oboseală)",
        "advice_low": "monitorizează evoluția și menține confortul interior stabil",
        "advice_high": "urmărește tendința; presiunea mare e frecvent asociată cu vreme stabilă",
    },
    "pm25": {
        "name": "PM2.5",
        "unit": "µg/m³",
        "low": "<10",
        "normal": "10-35",
        "high": ">55",
        "meaning": "particule foarte fine care pătrund adânc în plămâni",
        "effects": "valorile mari cresc riscul de iritații respiratorii și agravarea simptomelor la persoane sensibile",
        "advice_low": "nivelul este bun; menține ventilația și sursele de poluare sub control",
        "advice_high": "reduce expunerea, aerisește când exteriorul e mai curat și folosește filtru HEPA",
    },
    "pm1": {
        "name": "PM1",
        "unit": "µg/m³",
        "low": "<8",
        "normal": "8-25",
        "high": ">35",
        "meaning": "particule ultrafine cu diametru sub 1 µm",
        "effects": "pot ajunge foarte adânc în sistemul respirator",
        "advice_low": "nivel favorabil; continuă monitorizarea",
        "advice_high": "limitează sursele de fum/praf fin și folosește filtrare eficientă",
    },
    "pm10": {
        "name": "PM10",
        "unit": "µg/m³",
        "low": "<20",
        "normal": "20-50",
        "high": ">100",
        "meaning": "particule mai mari (praf, polen, particule de trafic)",
        "effects": "pot irita căile respiratorii și ochii la concentrații ridicate",
        "advice_low": "nivel bun; menține curățenia și ventilația",
        "advice_high": "aerisește în intervale cu trafic redus și folosește purificator",
    },
    "co2": {
        "name": "CO₂",
        "unit": "ppm",
        "low": "<400",
        "normal": "400-1000",
        "high": ">1500",
        "meaning": "indicator de ventilație; valori mari arată aer interior insuficient reîmprospătat",
        "effects": "CO₂ crescut poate da somnolență, scăderea concentrării și dureri de cap",
        "advice_low": "valoare foarte mică, de regulă asociată cu aer proaspăt abundent",
        "advice_high": "crește ventilația imediat (ferestre/ventilație mecanică)",
    },
    "voc": {
        "name": "VOC",
        "unit": "ppb (în majoritatea senzorilor)",
        "low": "<250",
        "normal": "250-600",
        "high": ">600",
        "meaning": "compuși organici volatili proveniți din vopsele, solvenți, mobilier, produse de curățare",
        "effects": "nivelurile ridicate pot provoca iritații, dureri de cap și disconfort",
        "advice_low": "nivel bun; menține sursele chimice sub control",
        "advice_high": "aerisește bine și reduce sursele de emisii VOC",
    },
    "lux": {
        "name": "Lux (iluminare)",
        "unit": "lx",
        "low": "<100",
        "normal": "100-500",
        "high": ">500",
        "meaning": "măsoară nivelul de iluminare din încăpere",
        "effects": "lumină prea mică poate obosi ochii, lumină prea puternică poate crea disconfort vizual",
        "advice_low": "crește iluminarea locală sau generală (lampă birou, lumină ambientală)",
        "advice_high": "reduce intensitatea sau folosește lumină difuză",
    },
    "st": {
        "name": "ST (temperatura senzor SCD41)",
        "unit": "°C",
        "low": "<18",
        "normal": "20-26",
        "high": ">28",
        "meaning": "temperatura măsurată direct de senzorul SCD41",
        "effects": "devierea față de confort poate indica disconfort termic local",
        "advice_low": "verifică amplasarea senzorului și crește temperatura ambientală dacă este nevoie",
        "advice_high": "evită sursele locale de căldură lângă senzor și îmbunătățește răcirea/ventilația",
    },
    "sh": {
        "name": "SH (umiditatea senzor SCD41)",
        "unit": "%",
        "low": "<30",
        "normal": "40-60",
        "high": ">70",
        "meaning": "umiditatea relativă măsurată direct de senzorul SCD41",
        "effects": "valorile extreme cresc disconfortul și pot afecta calitatea aerului percepută",
        "advice_low": "crește umiditatea treptat și evită uscarea excesivă a aerului",
        "advice_high": "aerisește și controlează sursele de umiditate",
    },
}

REFERENCE_ALIASES: dict[str, str] = {
    "pm1": "pm1",
    "pm 1": "pm1",
    "pm2.5": "pm25",
    "pm 2.5": "pm25",
    "pm25": "pm25",
    "pm10": "pm10",
    "pm 10": "pm10",
    "praf": "pm10",
    "praful": "pm10",
    "pulberi": "pm10",
    "co2": "co2",
    "co₂": "co2",
    "dioxid de carbon": "co2",
    "temperatura aerului": "temperature",
    "temperatura": "temperature",
    "temperature": "temperature",
    "umiditatea relativa": "humidity",
    "umiditate": "humidity",
    "humidity": "humidity",
    "presiune": "pressure",
    "presiunea atmosferica": "pressure",
    "hpa": "pressure",
    "voc": "voc",
    "tvoc": "voc",
    "compusi organici volatili": "voc",
    "lux": "lux",
    "iluminare": "lux",
    "lumina": "lux",
    "scd41": "st",
}

LIVE_FEATURE_ALIASES: dict[str, str] = {
    "pm 2.5": "pm25",
    "pm2.5": "pm25",
    "pm25": "pm25",
    "pm 10": "pm10",
    "pm10": "pm10",
    "praf": "pm10",
    "praful": "pm10",
    "pulberi": "pm10",
    "pm 1": "pm1",
    "pm1": "pm1",
    "co2": "co2",
    "co₂": "co2",
    "dioxid de carbon": "co2",
    "temperatur": "temperature",
    "umiditat": "humidity",
    "presiun": "pressure",
    "voc": "voc",
    "tvoc": "voc",
    "lux": "lux",
}

LIVE_MEASUREMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature", "temperatura"),
    "humidity": ("humidity", "umiditate"),
    "pressure": ("pressure", "presiune"),
    "pm1": ("pm1",),
    "pm25": ("pm25", "pm2_5"),
    "pm10": ("pm10",),
    "co2": ("co2",),
    "voc": ("voc", "tvoc"),
    "lux": ("lux",),
}

REFERENCE_CODE_ALIASES: dict[str, str] = {
    "t": "temperature",
    "h": "humidity",
    "p": "pressure",
    "st": "st",
    "sh": "sh",
}

# Phrases the user can use to explicitly teach the chatbot something to remember.
TEACHING_TRIGGER_PHRASES: tuple[str, ...] = (
    "ține minte", "tine minte",
    "reține că", "retine ca", "reține ca", "retine că",
    "memorează", "memoreaza",
    "notează că", "noteaza ca", "notează ca", "noteaza că",
    "să reții", "sa retii", "sa retin", "să rețin",
    "vreau să știi că", "vreau sa stii ca",
    "de acum încolo", "de acum incolo",
    "regulă:", "regula:",
)

# A conditional instruction like "daca X e deconectat, verifica Y" is also treated as
# something to learn, even without an explicit "ține minte" trigger phrase.
CONDITIONAL_INSTRUCTION_PATTERN = re.compile(
    r"dac[ăa]\b.{0,120}\b(verific\w*|check\w*|uit[ăa]\w*|resetea\w*|reporne\w*|schimb\w*)"
)

# Tokens that indicate the user is reporting a problem with a sensor/device, so any
# previously learned instructions about that situation should be recalled automatically.
SENSOR_PROBLEM_TOKENS: tuple[str, ...] = (
    "decuplat", "deconectat", "deconectata", "deconectată",
    "offline", "nu functioneaza", "nu funcționează", "nu merge",
    "nu raspunde", "nu răspunde", "senzor mort", "eroare senzor", "cablu",
)

_NOTE_STOPWORDS: frozenset[str] = frozenset({
    "daca", "dacă", "atunci", "pentru", "despre", "acest", "aceasta", "acesta",
    "este", "sunt", "care", "cand", "când", "unde", "cum", "sau", "din",
    "verifica", "verifică", "senzor", "senzorul", "senzorii", "device", "dispozitiv",
})


@dataclass
class ChatbotContext:
    """Container for context that can be extended with ML outputs and fresh data."""

    latest_measurement: dict[str, Any] | None = None
    model_outputs: dict[str, Any] | None = None
    selected_device_identifier: str | None = None
    available_devices: list[dict[str, Any]] | None = None
    learned_notes: list[dict[str, Any]] | None = None


class RuleBasedAirQualityChatbot:
    """Simple rule-based chatbot designed to be replaceable with an LLM later."""

    def build_context(
        self,
        model_outputs: dict[str, Any] | None = None,
        device_identifier: str | None = None,
    ) -> ChatbotContext:
        latest_measurement = None
        location_details = None
        available_devices: list[dict[str, Any]] = []

        # The API layer remains resilient even if database access fails.
        with suppress(RuntimeError, ValueError, KeyError, TypeError):
            # Chat needs only the newest row; avoid loading the full table on each message.
            df = get_measurements(
                limit=1,
                descending=True,
                device_identifier=device_identifier,
            )
            if not df.empty:
                latest_measurement = self._extract_latest_measurement(df)

        with suppress(RuntimeError, ValueError, KeyError, TypeError):
            location_aliases: list[str] = []
            if latest_measurement is not None:
                for key in ("device_id", "deviceId", "sensor_id", "sensorId", "device_name", "name"):
                    raw_value = latest_measurement.get(key)
                    if raw_value is None:
                        continue
                    value = str(raw_value).strip()
                    if value and value not in location_aliases:
                        location_aliases.append(value)

            lookup_identifier = device_identifier
            if lookup_identifier is None and latest_measurement is not None:
                measurement_identifier = str(latest_measurement.get("device_identifier") or "").strip()
                lookup_identifier = measurement_identifier or None

            if lookup_identifier:
                location_details = get_device_location_details(lookup_identifier, aliases=location_aliases)

        with suppress(RuntimeError, ValueError, KeyError, TypeError):
            available_devices = get_devices_with_location()

        if location_details:
            if latest_measurement is None:
                latest_measurement = {"device_identifier": device_identifier}
            latest_measurement["location"] = location_details.get("location")
            latest_measurement["latitude"] = location_details.get("latitude")
            latest_measurement["longitude"] = location_details.get("longitude")
            latest_measurement["location_source_table"] = location_details.get("source_table")
            latest_measurement["location_source_column"] = location_details.get("source_id_column")

        return ChatbotContext(
            latest_measurement=latest_measurement,
            model_outputs=model_outputs or {},
            selected_device_identifier=device_identifier,
            available_devices=available_devices,
        )

    @staticmethod
    def is_devices_question(normalized_message: str) -> bool:
        device_markers = [
            "toate dispozitivele",
            "ce dispozitive",
            "lista dispozitive",
            "dispozitivele",
            "device-uri",
            "deviceuri",
            "senzori disponibili",
        ]
        return any(token in normalized_message for token in device_markers)

    @staticmethod
    def is_capabilities_question(normalized_message: str) -> bool:
        capability_markers = [
            "toate functiile",
            "toate funcțiile",
            "ce functii",
            "ce funcții",
            "ce poti face",
            "ce poți face",
            "ce stii sa faci",
            "ce știi să faci",
        ]
        return any(token in normalized_message for token in capability_markers)

    @staticmethod
    def is_teaching_message(normalized_message: str) -> bool:
        if any(phrase in normalized_message for phrase in TEACHING_TRIGGER_PHRASES):
            return True
        return bool(CONDITIONAL_INSTRUCTION_PATTERN.search(normalized_message))

    @staticmethod
    def extract_taught_note(original_message: str, normalized_message: str) -> str:
        for phrase in TEACHING_TRIGGER_PHRASES:
            index = normalized_message.find(phrase)
            if index != -1:
                remainder = original_message[index + len(phrase):].strip(" :,-")
                return remainder or original_message.strip()
        return original_message.strip()

    @staticmethod
    def _note_keywords(text: str) -> set[str]:
        words = re.findall(r"[a-zăâîșț0-9]+", text.lower())
        return {word for word in words if len(word) >= 4 and word not in _NOTE_STOPWORDS}

    @staticmethod
    def find_relevant_notes(
        normalized_message: str,
        notes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        message_keywords = RuleBasedAirQualityChatbot._note_keywords(normalized_message)
        if not message_keywords:
            return []

        relevant: list[dict[str, Any]] = []
        for note in notes:
            note_text = str(note.get("note") or "")
            note_keywords = RuleBasedAirQualityChatbot._note_keywords(note_text)
            if message_keywords & note_keywords:
                relevant.append(note)
        return relevant

    @staticmethod
    def is_location_question(normalized_message: str) -> bool:
        location_markers = [
            "unde",
            "unde e",
            "unde este",
            "se afla",
            "se află",
            "locatie",
            "locație",
            "amplasat",
            "pozitie",
            "poziție",
            "coordonate",
            "gps",
            "latitudine",
            "longitudine",
            "oras",
            "oraș",
            "adresa",
            "adresă",
        ]
        device_markers = [
            "dispozitiv",
            "dispoz",
            "dispzo",
            "device",
            "senzor",
            "camera",
            "cameră",
        ]
        has_location_marker = any(token in normalized_message for token in location_markers)
        has_device_marker = any(token in normalized_message for token in device_markers)
        return has_location_marker and has_device_marker

    @staticmethod
    def has_location_markers(normalized_message: str) -> bool:
        return any(
            token in normalized_message
            for token in ["unde", "locatie", "locație", "coordonate", "gps", "latitudine", "longitudine"]
        )

    @staticmethod
    def is_forecast_question(normalized_message: str) -> bool:
        forecast_markers = ["peste", "ore", "ora", "forecast", "prognoza", "prognoză", "va fi", "cat va fi", "cât va fi"]
        return any(token in normalized_message for token in forecast_markers)

    @staticmethod
    def _extract_forecast_horizon_hours(normalized_message: str) -> int | None:
        match = re.search(r"peste\s+(\d{1,3})\s*(?:ore|ora|h)", normalized_message)
        if match:
            return max(1, int(match.group(1)))

        match = re.search(r"(?:in|în)\s+(\d{1,3})\s*(?:ore|ora|h)", normalized_message)
        if match:
            return max(1, int(match.group(1)))

        return None

    @staticmethod
    def _extract_forecast_feature(normalized_message: str) -> str | None:
        feature_aliases = {
            "temperature": ["temperatura", "temperatură", "temperature", "temp"],
            "humidity": ["umiditate", "humidity"],
            "pm25": ["pm2.5", "pm25", "pm 2.5"],
            "pm10": ["pm10", "pm 10", "praf"],
            "co2": ["co2", "co₂", "dioxid"],
        }
        for feature_name, aliases in feature_aliases.items():
            if any(alias in normalized_message for alias in aliases):
                return feature_name
        return None

    @staticmethod
    def _extract_base_feature_values_from_measurement(measurement: dict[str, Any]) -> dict[str, float] | None:
        feature_fields = {
            "temperature": ("temperature", "temperatura", "temp"),
            "humidity": ("humidity", "umiditate"),
            "pm25": ("pm25", "pm2_5", "pm2.5"),
            "pm10": ("pm10",),
            "co2": ("co2", "co_2"),
        }

        base_values: dict[str, float] = {}
        for feature_name, candidates in feature_fields.items():
            found_value = None
            for candidate in candidates:
                if candidate in measurement and measurement[candidate] is not None and str(measurement[candidate]).strip() != "":
                    try:
                        found_value = float(measurement[candidate])
                        break
                    except (TypeError, ValueError):
                        continue
            if found_value is None:
                return None
            base_values[feature_name] = found_value

        return base_values

    @staticmethod
    def is_live_data_question(normalized_message: str) -> bool:
        current_markers = [
            "acum", "actual", "actuala", "camera mea", "in camera", "în cameră",
            "ultima măsurătoare", "ultima masuratoare", "ce temperatura", "ce temperatură",
            "ce valori", "valoarea", "valorile", "nivelul", "cât", "cat ",
            "calitatea aerului", "starea aerului", "etichet", "predic", "prognoz",
        ]
        if not any(token in normalized_message for token in current_markers):
            return False
        has_sensor = any(token in normalized_message for token in LIVE_FEATURE_ALIASES)
        return has_sensor or bool(detect_sensor_features(normalized_message))

    # Any of these intents need a live prediction to answer properly, so the
    # chatbot should fetch one automatically instead of asking the user to run it.
    _PREDICTION_TRIGGER_TOKENS = (
        "anomal", "anomalie", "ciudat", "neobișnuit",
        "pm2.5", "pm25", "pm10", "particule",
        "co2", "dioxid", "ventilatie", "ventilație", "aer proaspat",
        "predict", "predic", "forecast", "prognoza", "prognoză", "stare",
        "recomand", "recomandare", "ce trebuie", "ar trebui",
    )

    @classmethod
    def needs_live_prediction(cls, normalized_message: str) -> bool:
        return any(token in normalized_message for token in cls._PREDICTION_TRIGGER_TOKENS)

    @staticmethod
    def _requested_live_features(normalized_message: str) -> list[str]:
        features: list[str] = []
        for alias, feature in sorted(LIVE_FEATURE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if alias in normalized_message and feature not in features:
                features.append(feature)
        if not features:
            features.extend(detect_sensor_features(normalized_message))
        return features

    @staticmethod
    def try_build_live_prediction(
        model_outputs: dict[str, Any],
        device_identifier: str | None = None,
    ) -> None:
        existing_prediction = model_outputs.get("latest_prediction")
        if existing_prediction and existing_prediction.get("device_identifier") == device_identifier:
            return
        try:
            prediction, confidence, feature_values, feature_assessment = predict_air_quality(
                device_identifier=device_identifier,
            )
        except (RuntimeError, ValueError, KeyError, TypeError, OSError):
            return
        model_outputs["latest_prediction"] = {
            "prediction": str(prediction),
            "confidence": confidence,
            "input_values": feature_values,
            "feature_assessment": feature_assessment,
            "device_identifier": device_identifier,
        }

    def generate_reply(
        self,
        message: str,
        context: ChatbotContext,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        normalized = (message or "").strip().lower()

        if not normalized:
            return (
                "Te pot ajuta cu informații despre calitatea aerului. "
                "Întreabă-mă despre predicții, anomalii, modele, PM2.5, CO2 sau cum interpret măsurătorile."
            )

        if any(token in normalized for token in ["salut", "hello", "buna", "bună"]):
            return (
                f"{WELCOME_MESSAGE} "
                "Vreau să știu cu ce te pot ajuta."
            )

        if any(token in normalized for token in SENSOR_PROBLEM_TOKENS):
            relevant_notes = self.find_relevant_notes(normalized, context.learned_notes or [])
            if relevant_notes:
                tips = "; ".join(str(note.get("note")) for note in relevant_notes[:3])
                return (
                    f"Mi-ai spus anterior: {tips}. "
                    "Verifică asta acum; spune-mi dacă problema persistă."
                )

        if self.is_location_question(normalized) or (
            context.selected_device_identifier is not None and self.has_location_markers(normalized)
        ):
            return self._format_device_location_reply(context)

        if self.is_forecast_question(normalized):
            forecast_reply = self._format_forecast_question_reply(
                context, normalized, conversation_history=conversation_history
            )
            if forecast_reply:
                return forecast_reply

        if self.is_devices_question(normalized):
            return self._format_devices_catalog_reply(context)

        if self.is_capabilities_question(normalized):
            return self._format_capabilities_reply(context)

        if self.is_live_data_question(normalized):
            return self._format_live_status_reply(context, normalized)

        if any(token in normalized for token in ["ug/m3", "µg/m3", "µg/m³", "ppm", "ce inseamna", "ce înseamnă", "unitate", "prag", "interval"]):
            reference_reply = self.build_reference_reply(message, normalized)
            if reference_reply:
                return reference_reply

        if any(token in normalized for token in ["anomal", "anomalie", "ciudat", "neobișnuit"]):
            model_outputs = context.model_outputs or {}
            latest_prediction = model_outputs.get("latest_prediction")
            if latest_prediction:
                return self._analyze_anomalies_from_prediction(latest_prediction)
            return (
                "Pentru a detecta anomalii, te rog să rulezi mai întâi o predicție. "
                "Apoi voi analiza valorile și voi identifica orice comportamente neobișnuite."
            )

        if any(token in normalized for token in ["pm2.5", "pm25", "pm10", "particule"]):
            model_outputs = context.model_outputs or {}
            latest_prediction = model_outputs.get("latest_prediction")
            if latest_prediction:
                assessment = latest_prediction.get("feature_assessment") or {}
                pm25_data = assessment.get("pm25") or {}
                pm10_data = assessment.get("pm10") or {}
                
                pm25_val = pm25_data.get("value", "n/a")
                pm10_val = pm10_data.get("value", "n/a")
                pm25_status = pm25_data.get("status", "necunoscut")
                pm10_status = pm10_data.get("status", "necunoscut")
                
                return (
                    f"PM2.5 este la {pm25_val} µg/m³ ({pm25_status}), iar PM10 este la {pm10_val} µg/m³ ({pm10_status}). "
                    f"Particulele fine afectează direct respirația. Dacă valorile sunt ridicate, "
                    f"recomand filtrare și limitarea expunerii la exterior."
                )
            # Provide educational info without predictions
            tips = [
                "PM2.5 și PM10 sunt particule fine și grosiere din aer. Valorile sub 35 µg/m³ (PM2.5) sunt considerate bune. Dacă nu ai predicție, rulează Predict din tab-ul corespunzător.",
                "Particulele fine (PM2.5) pătrund adânc în plămâni și afectează respirația. Utilizează filtre HEPA dacă valorile sunt ridicate și aerisește des.",
                "PM10 sunt particule mai mari din praf, polen și poluare. PM2.5 sunt mai periculoase pentru sănătate. Ai vrea să rulez o predicție pentru valorile actuale?"
            ]
            import random
            return random.choice(tips)

        if any(token in normalized for token in ["co2", "dioxid", "ventilatie", "ventilație", "aer proaspat"]):
            model_outputs = context.model_outputs or {}
            latest_prediction = model_outputs.get("latest_prediction")
            if latest_prediction:
                assessment = latest_prediction.get("feature_assessment") or {}
                co2_data = assessment.get("co2") or {}
                
                co2_val = co2_data.get("value", "n/a")
                co2_status = co2_data.get("status", "necunoscut")
                
                return (
                    f"CO2 este la {co2_val} ppm ({co2_status}). "
                    f"Dacă este ridicat, indică ventilație slabă și disconfort. "
                    f"Soluția este deschiderea ferestrelor sau crește debitul de aer proaspăt."
                )
            # Provide varied tips without predictions
            import random
            co2_tips = [
                "Nivelul CO2 normal este sub 1000 ppm. Peste 1500 ppm indică ventilație slabă. Deschide ferestre și ventilează regulat!",
                "CO2 accumuleaza în încăperi fără ventilație. Soluție simplă: deschide ferestre 5-10 minute sau pornește ventilatorul.",
                "Aerul proaspăt reduce CO2 și crește confortul. Recomandare: ventilație încrucișată. Pentru valori actuale, rulează Predict.",
                "CO2 ridicat cauzeaza dureri de cap și oboseală. Soluția imediată: aerisire. Vrei o predicție pentru datele actuale?"
            ]
            return random.choice(co2_tips)

        if any(token in normalized for token in ["predict", "predic", "forecast", "prognoza", "prognoză", "stare"]):
            prediction_payload = (context.model_outputs or {}).get("latest_prediction")
            if not prediction_payload:
                import random
                predict_tips = [
                    "Nu am încă o predicție. Rulează Predict din tab-ul corespunzător pentru a obține o evaluare cu informații detaliate.",
                    "Fără predicție, nu pot analiza datele. Click pe 'Predict' și voi procesa valorile senzorilor.",
                    "Nu s-a rulat încă vreo predicție. Mergi la tab-ul 'Predict' și apasă butonul pentru a-mi aduce date reale.",
                ]
                return random.choice(predict_tips)
            reply_text = self._format_prediction_reply(prediction_payload)
            return reply_text + self._device_scope_note(prediction_payload.get("device_identifier"))

        if any(token in normalized for token in ["training", "train", "antren", "model", "oob", "accuracy", "f1", "performant"]):
            training_payload = (context.model_outputs or {}).get("latest_training")
            if not training_payload:
                import random
                train_tips = [
                    "Nu am rezultate de antrenare disponibile. Rulează Train din tab-ul corespunzător pentru a evalua performanța modelului.",
                    "Modelul nu a fost antrenat încă. Click pe 'Train' și voi afișa metrica de acuratețe și alți parametri.",
                    "Fără un training recent, nu pot raporta performanța modelului. Du-te la tab-ul 'Train' și antrenează modelul.",
                ]
                return random.choice(train_tips)
            return self._format_training_reply(training_payload)

        if any(token in normalized for token in ["ultima", "ultime", "masuratoare", "măsurătoare", "valori actuale"]):
            if context.latest_measurement is None:
                return (
                    "Nu sunt măsurători disponibile în baza de date. "
                    "Verifică senzorul și conectarea la Supabase."
                )
            return self._format_latest_measurement_reply(context.latest_measurement)

        if any(token in normalized for token in ["recomand", "recomandare", "ce trebuie", "cum", "ar trebui"]):
            model_outputs = context.model_outputs or {}
            latest_prediction = model_outputs.get("latest_prediction")
            if latest_prediction:
                return self._generate_recommendations(latest_prediction)
            import random
            rec_tips = [
                "Pentru recomandări personalizate, rulează mai întâi Predict. Voi analiza valorile și voi oferi sfaturi bazate pe datele actuale.",
                "Recomandări? Am nevoie de o predicție mai întâi! Click pe 'Predict' și apoi mă întrebi din nou.",
                "Nu pot face recomandări fără date. Rulează Predict și voi genera sfaturi specifice pentru mediul tău.",
            ]
            return random.choice(rec_tips)

        if any(token in normalized for token in ["ajutor", "help", "ce poti", "ce poți", "comenzi"]):
            return (
                "Pot ajuta cu: 🔹 Predicții calitate aer 🔹 Anomalii și alerte 🔹 CO2/PM2.5/PM10 🔹 Recomandări de măsuri "
                "🔹 Performanța modelor. Întreabă: 'Ce spui despre PM2.5?', 'Cum e predicția?', 'Ce recomandari ai?'"
            )

        # Generic fallback with rotating tips
        fallback_tips = [
            "Nu sunt sigur că am înțeles. Încearcă: 'Spune-mi despre PM2.5', 'Ce anomalii?', 'Cum e predicția?', 'Ce recomandari?'",
            "Nteles! Pentru răspunsuri precise, rulează mai întâi o predicție din tab-ul 'Predict', apoi voi putea analiza cu date reale.",
            "Nu am găsit pattern-ul. Poti să mă întrebi despre: anomalii, PM particule, CO2, predicții, training, ultima măsurătoare, recomandări.",
            "Încearcă să reformulezi. De exemplu: 'Cum e calitatea aerului?' este mai vag decât 'Spune-mi despre PM2.5'."
        ]
        import random
        return random.choice(fallback_tips)

    @staticmethod
    def _analyze_anomalies_from_prediction(prediction_payload: dict[str, Any]) -> str:
        """Analyze and describe anomalies detected in the latest prediction."""
        assessment = prediction_payload.get("feature_assessment") or {}
        
        if not assessment:
            return "Nu am date suficiente pentru a detecta anomalii în acest moment."
        
        issues = []
        for feature_name, details in assessment.items():
            if isinstance(details, dict):
                status = details.get("status", "").lower()
                value = details.get("value", "n/a")
                reason = details.get("reason", "")
                
                if "poor" in status or "warning" in status or details.get("sensor_warning"):
                    unit = details.get("unit", "")
                    issues.append(f"{feature_name}: {value} {unit} ({reason})" if reason else f"{feature_name}: {value} {unit}")
        
        if not issues:
            return "Analizând datele, nu am detectat anomalii semnificative. Calitatea aerului pare normală."
        
        issue_text = "; ".join(issues)
        return f"Am detectat probleme: {issue_text}. Te recomand să iei măsuri corective, cum ar fi aerisire sau filtrare."

    @staticmethod
    def _generate_recommendations(prediction_payload: dict[str, Any]) -> str:
        """Generate actionable recommendations based on prediction data."""
        prediction = prediction_payload.get("prediction", "unknown")
        assessment = prediction_payload.get("feature_assessment") or {}
        
        recommendations = []
        
        # Check PM2.5
        pm25_data = assessment.get("pm25") or {}
        if pm25_data.get("status") == "poor":
            recommendations.append("Stare alarmă: Nivelul PM2.5 este critic. Rămâi în casă și folosește filtre HEPA.")
        elif pm25_data.get("status") == "moderate":
            recommendations.append("PM2.5 este moderat ridicat. Aeriseste periodic și poartă mască dacă ieși.")
        
        # Check CO2
        co2_data = assessment.get("co2") or {}
        if co2_data.get("status") == "poor":
            recommendations.append("CO2 este prea ridicat. Deschide imediat ferestrele pentru ventilație.")
        elif co2_data.get("status") == "moderate":
            recommendations.append("Ventilația poate fi îmbunătățită. Lasă ferestrele deschise pentru aer proaspăt.")
        
        # Check Temperature
        temp_data = assessment.get("temperature") or {}
        if temp_data.get("status") == "poor":
            recommendations.append("Temperatura nu este confortabilă. Reglează încălzirea sau răcirea.")
        
        # Check Humidity
        humid_data = assessment.get("humidity") or {}
        if humid_data.get("status") == "poor":
            recommendations.append("Umiditatea este în afara intervalului normal. Folosește umidificator sau deumidificator.")
        
        if not recommendations:
            quality_msg = f"Calitatea aerului este {prediction}. Condiții normale - continuă monitorizarea."
            return quality_msg
        
        return " ".join(recommendations)

    @staticmethod
    def _extract_latest_measurement(df: pd.DataFrame) -> dict[str, Any]:
        if "created_at" in df.columns:
            latest_row = df.sort_values("created_at", ascending=False).iloc[0]
        else:
            latest_row = df.iloc[-1]

        return latest_row.to_dict()

    @staticmethod
    def _format_latest_measurement_reply(measurement: dict[str, Any]) -> str:
        co2 = measurement.get("co2", "n/a")
        pm25 = measurement.get("pm25", "n/a")
        pm10 = measurement.get("pm10", "n/a")
        temperature = measurement.get("temperature", "n/a")
        humidity = measurement.get("humidity", "n/a")

        return (
            "Ultima măsurătoare disponibilă este: "
            f"temperatură={temperature}, umiditate={humidity}, CO2={co2}, PM2.5={pm25}, PM10={pm10}."
        )

    @staticmethod
    def _extract_location_details(measurement: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
        text_candidates = ["location", "location_name", "locatie", "city", "address", "device_location"]
        lat_candidates = ["latitude", "lat", "gps_lat", "device_latitude"]
        lon_candidates = ["longitude", "lon", "lng", "gps_lng", "device_longitude"]

        location_text = None
        for key in text_candidates:
            value = measurement.get(key)
            if value is not None and str(value).strip() != "":
                location_text = str(value).strip()
                break

        latitude = None
        for key in lat_candidates:
            value = measurement.get(key)
            if value is None or str(value).strip() == "":
                continue
            try:
                latitude = float(value)
                break
            except (TypeError, ValueError):
                continue

        longitude = None
        for key in lon_candidates:
            value = measurement.get(key)
            if value is None or str(value).strip() == "":
                continue
            try:
                longitude = float(value)
                break
            except (TypeError, ValueError):
                continue

        return location_text, latitude, longitude

    @staticmethod
    def _format_device_location_reply(context: ChatbotContext) -> str:
        measurement = context.latest_measurement
        selected_device = context.selected_device_identifier

        if measurement is None:
            if selected_device:
                return (
                    f"Nu am găsit măsurători recente pentru dispozitivul '{selected_device}', "
                    "deci nu pot determina locația momentan."
                )
            return "Nu am găsit măsurători recente, deci nu pot determina locația dispozitivului."

        device_identifier = str(measurement.get("device_identifier") or selected_device or "dispozitivul selectat").strip()
        location_text, latitude, longitude = RuleBasedAirQualityChatbot._extract_location_details(measurement)

        if location_text and latitude is not None and longitude is not None:
            return (
                f"Dispozitivul '{device_identifier}' este în '{location_text}', "
                f"la coordonatele {latitude:.5f}, {longitude:.5f}."
            )

        if location_text:
            return f"Dispozitivul '{device_identifier}' este în '{location_text}'."

        if latitude is not None and longitude is not None:
            return (
                f"Dispozitivul '{device_identifier}' are coordonatele "
                f"{latitude:.5f}, {longitude:.5f}."
            )

        return (
            f"Pentru dispozitivul '{device_identifier}' nu am încă informații de locație "
            "(nici text, nici coordonate GPS)."
        )

    @staticmethod
    def _find_forecast_feature_in_history(conversation_history: list[dict[str, str]] | None) -> str | None:
        """Recall the last forecast/live feature the user mentioned, for follow-up questions like 'dar peste 3 ore?'."""
        for history_message in reversed((conversation_history or [])[-12:]):
            if history_message.get("role") != "user":
                continue
            content = (history_message.get("content") or "").strip().lower()
            if not content:
                continue
            feature = RuleBasedAirQualityChatbot._extract_forecast_feature(content)
            if feature is not None:
                return feature
        return None

    @staticmethod
    def _format_forecast_question_reply(
        context: ChatbotContext,
        normalized_message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str | None:
        horizon_hours = RuleBasedAirQualityChatbot._extract_forecast_horizon_hours(normalized_message)
        if horizon_hours is None:
            return None

        requested_feature = RuleBasedAirQualityChatbot._extract_forecast_feature(normalized_message)
        if requested_feature is None:
            requested_feature = RuleBasedAirQualityChatbot._find_forecast_feature_in_history(conversation_history)
        if requested_feature is None:
            return (
                "Pot calcula prognoza, dar spune-mi și parametrul dorit (ex: temperatură, umiditate, PM2.5, PM10, CO2)."
            )

        measurement = context.latest_measurement
        if measurement is None:
            return "Nu am o măsurătoare curentă pentru a calcula prognoza."

        base_values = RuleBasedAirQualityChatbot._extract_base_feature_values_from_measurement(measurement)
        if base_values is None:
            return "Nu am toate valorile de bază necesare pentru prognoză (temperatură, umiditate, PM2.5, PM10, CO2)."

        try:
            forecast = build_forecast(
                base_feature_values=base_values,
                horizons_hours=[horizon_hours],
                lookback_hours=max(24, min(720, horizon_hours * 4)),
                device_identifier=context.selected_device_identifier,
            )
        except RuntimeError as exc:
            return f"Nu pot calcula prognoza acum: {exc}"
        except Exception:
            return "A apărut o eroare la calculul prognozei pentru întrebarea ta."

        if not forecast:
            return "Nu am putut genera prognoza pentru intervalul cerut."

        first_point = forecast[0]
        values = first_point.get("input_values") or {}
        if requested_feature not in values:
            return "Prognoza a fost generată, dar parametrul cerut nu este disponibil în rezultat."

        feature_name = {
            "temperature": "temperatura",
            "humidity": "umiditatea",
            "pm25": "PM2.5",
            "pm10": "PM10",
            "co2": "CO2",
        }[requested_feature]
        feature_unit = {
            "temperature": "°C",
            "humidity": "%",
            "pm25": "µg/m³",
            "pm10": "µg/m³",
            "co2": "ppm",
        }[requested_feature]

        predicted_value = float(values[requested_feature])
        quality_prediction = first_point.get("prediction")
        confidence = first_point.get("confidence")
        confidence_text = ""
        if isinstance(confidence, (int, float)):
            confidence_text = f" (încredere {float(confidence):.1%})"

        return (
            f"Peste {horizon_hours} ore, estimarea pentru {feature_name} este {predicted_value:.2f} {feature_unit}. "
            f"Predicția generală a calității aerului: {quality_prediction}{confidence_text}."
            f"{RuleBasedAirQualityChatbot._device_scope_note(context.selected_device_identifier)}"
        )

    @staticmethod
    def _format_devices_catalog_reply(context: ChatbotContext) -> str:
        devices = context.available_devices or []
        if not devices:
            return "Nu am găsit dispozitive pe contul curent sau nu am acces la ele în acest moment."

        lines = []
        for item in devices[:20]:
            identifier = str(item.get("device_identifier") or "dispozitiv-necunoscut")
            location_label = str(item.get("location_label") or "Locație necunoscută")
            lines.append(f"- {identifier}: {location_label}")

        suffix = ""
        if len(devices) > 20:
            suffix = f"\n... și încă {len(devices) - 20} dispozitive."

        return f"Am găsit {len(devices)} dispozitive pe cont:\n" + "\n".join(lines) + suffix

    @staticmethod
    def _format_capabilities_reply(context: ChatbotContext) -> str:
        devices_count = len(context.available_devices or [])
        return (
            "Pot acoperi funcțiile principale din aplicație: predicție curentă, prognoză, "
            "detecție anomalii, explicații indicatori (PM/CO2/temperatură/umiditate), "
            "detalii despre antrenare și locația dispozitivelor. "
            f"În prezent văd {devices_count} dispozitive în contul tău."
        )

    @staticmethod
    def _format_live_status_reply(context: ChatbotContext, normalized_message: str) -> str:
        measurement = context.latest_measurement
        if measurement is None:
            return "Nu am găsit o măsurătoare actuală în baza de date Supabase. Verifică senzorul și conexiunea."

        requested_features = RuleBasedAirQualityChatbot._requested_live_features(normalized_message)
        if not requested_features or "calitatea aerului" in normalized_message or "starea aerului" in normalized_message:
            requested_features = ["temperature", "humidity", "pm25", "pm10", "co2"]

        feature_names = {
            "temperature": ("temperatura", "°C"),
            "humidity": ("umiditatea", "%"),
            "pressure": ("presiunea", "hPa"),
            "pm1": ("PM1", "µg/m³"),
            "pm25": ("PM2.5", "µg/m³"),
            "pm10": ("PM10 (praf)", "µg/m³"),
            "co2": ("CO2", "ppm"),
            "voc": ("VOC", "ppb"),
            "lux": ("lumina", "lx"),
        }
        values = []
        for feature in requested_features:
            field_names = LIVE_MEASUREMENT_FIELDS.get(feature, (feature,))
            value = next((measurement[field] for field in field_names if field in measurement), "n/a")
            name, unit = feature_names[feature]
            values.append(f"{name}={value} {unit}")

        quality_label = str(measurement.get("quality_label") or "").strip().lower()
        label_names = {"good": "bună", "moderate": "moderată", "poor": "slabă"}
        prediction = (context.model_outputs or {}).get("latest_prediction") or {}
        prediction_label = str(prediction.get("prediction") or "").strip().lower()
        details = []
        if quality_label in label_names:
            details.append(f"eticheta din baza de date este «{label_names[quality_label]}»")
        if prediction_label in label_names:
            details.append(f"predicția modelului indică o calitate {label_names[prediction_label]}")
        suffix = f" {'; '.join(details).capitalize()}." if details else " Nu există încă o etichetă validă pentru această înregistrare."
        return f"Din ultima măsurătoare din baza de date: {'; '.join(values)}.{suffix}"

    @staticmethod
    def _device_scope_note(device_identifier: str | None) -> str:
        if device_identifier:
            return ""
        return (
            " (Nu ai specificat un dispozitiv anume, așa că am folosit datele agregate de pe toate dispozitivele."
            " Spune-mi identificatorul unui dispozitiv dacă vrei rezultatul doar pentru acela.)"
        )

    @staticmethod
    def _format_prediction_reply(prediction_payload: dict[str, Any]) -> str:
        prediction = prediction_payload.get("prediction", "necunoscut")
        confidence = prediction_payload.get("confidence")
        assessment = prediction_payload.get("feature_assessment") or {}
        
        # Build confidence text
        confidence_text = "n/a"
        if isinstance(confidence, (int, float)):
            confidence_text = f"{float(confidence):.1%}"
        
        # Analyze key metrics
        key_values = []
        for feature in ["temperature", "humidity", "co2", "pm25", "pm10"]:
            if feature in assessment:
                details = assessment[feature]
                if isinstance(details, dict):
                    value = details.get("value", "n/a")
                    unit = details.get("unit", "")
                    status = details.get("status", "")
                    key_values.append(f"{feature}={value}{unit} ({status})")
        
        values_text = ", ".join(key_values) if key_values else "date incomplete"
        
        # Count warnings
        warning_count = sum(
            1 for details in assessment.values()
            if isinstance(details, dict) and details.get("sensor_warning")
        )
        
        warning_text = ""
        if warning_count > 0:
            warning_text = f" ⚠️ {warning_count} avertizare(ări) de senzor!"
        
        # Forecast info
        forecast = prediction_payload.get("forecast") or []
        forecast_text = ""
        if isinstance(forecast, list) and len(forecast) > 0:
            horizons = [str(item.get("horizon_hours")) for item in forecast if item.get("horizon_hours")]
            if horizons:
                forecast_text = f" Prognoza pentru următoarele ore: +{', +'.join(horizons)}."
        
        # Generate natural sentence
        quality_descriptions = {
            "good": "bună",
            "moderate": "moderată",
            "poor": "proastă"
        }
        quality_desc = quality_descriptions.get(prediction, prediction)
        
        return (
            f"Predicția indică o calitate a aerului {quality_desc}, cu o încredere de {confidence_text}. "
            f"Valorile actuale: {values_text}.{warning_text}{forecast_text}"
        )

    @staticmethod
    def _format_training_reply(training_payload: dict[str, Any]) -> str:
        model_type = training_payload.get("model_type", "necunoscut")
        training_report = training_payload.get("training_report") or {}
        summary = training_report.get("summary") or {}
        dataset_info = training_report.get("dataset_info") or {}
        evaluation = training_report.get("evaluation") or {}
        anomaly_summary = training_report.get("anomaly_summary") or {}

        # Model info
        model_names = {
            "random_forest": "Random Forest",
            "isolation_forest": "Isolation Forest"
        }
        model_name = model_names.get(model_type, model_type)

        # Dataset info
        rows = dataset_info.get("row_count", "n/a")
        
        # Aggregation info
        agg_granularity = summary.get("aggregation_granularity")
        agg_value = summary.get("aggregation_value")
        if agg_granularity == "minute" and agg_value is not None:
            agg_text = f"{int(agg_value)} minute"
        elif agg_value is not None:
            agg_text = f"{int(agg_value)} ore"
        else:
            agg_text = "interval implicit"

        # Performance metrics
        accuracy = evaluation.get("accuracy")
        precision = evaluation.get("precision")
        recall = evaluation.get("recall")
        f1 = evaluation.get("f1_score")

        metrics_text = ""
        if model_type == "random_forest" and isinstance(accuracy, (int, float)):
            metrics_text = (
                f"Performanța: Acuratețe={accuracy:.1%}, Precizie={precision:.1%}, Recall={recall:.1%}, F1={f1:.1%}."
            )
        elif model_type == "isolation_forest":
            anomaly_count = anomaly_summary.get("anomaly_count", 0)
            anomaly_pct = anomaly_summary.get("anomaly_percentage", 0)
            metrics_text = f"A detectat {anomaly_count} anomalii ({anomaly_pct:.1f}% din date)."

        return (
            f"Antrenamentul cu {model_name} s-a finalizat pe {rows} înregistrări, "
            f"agregate la {agg_text}. {metrics_text}"
        )

    def build_reference_reply(self, original_message: str, normalized_message: str) -> str | None:
        detected_topics = self.detect_reference_topics(normalized_message)
        numeric_value = self._extract_numeric_value(normalized_message)

        if not detected_topics and any(token in normalized_message for token in ["ug/m3", "µg/m3", "µg/m³", "ppm"]):
            detected_topics = ["pm25", "pm10", "co2"]

        if detected_topics:
            details: list[str] = []
            for topic in detected_topics:
                details.append(self._build_parameter_brief(topic, measured_value=numeric_value))

            if len(details) == 1:
                return details[0]

            intro = "Am găsit mai mulți indicatori în întrebare. Pe scurt:"
            joined = " ".join(details)
            return f"{intro} {joined}"

        if not CHATBOT_ENABLE_WEB_SEARCH:
            return None

        query = self._extract_concept_query(original_message, normalized_message)
        if not query:
            return None

        web_summary = self._search_web_summary(query)
        if not web_summary:
            return None

        return (
            f"Nu am definitia in baza locala pentru '{query}', dar am gasit asta: {web_summary} "
            "Daca vrei, pot sa o traduc si in termeni mai practici pentru senzorii tai."
        )

    @staticmethod
    def detect_reference_topics(normalized_message: str) -> list[str]:
        topics: list[str] = []
        for alias, topic in REFERENCE_ALIASES.items():
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            if re.search(pattern, normalized_message) and topic not in topics:
                topics.append(topic)

        for code, topic in REFERENCE_CODE_ALIASES.items():
            pattern = rf"(?<![a-z0-9]){re.escape(code)}(?![a-z0-9])"
            if re.search(pattern, normalized_message) and topic not in topics:
                topics.append(topic)

        if re.search(r"(?<![a-z0-9])pm(?![a-z0-9]|\s*\d)", normalized_message):
            for topic in ("pm25", "pm10"):
                if topic not in topics:
                    topics.append(topic)

        return topics

    @staticmethod
    def _extract_numeric_value(normalized_message: str) -> float | None:
        sanitized_message = re.sub(
            r"\b(pm\s*2\.5|pm25|pm\s*10|pm10|pm\s*1|pm1|co2|co₂|scd41|st|sh)\b",
            " ",
            normalized_message,
            flags=re.IGNORECASE,
        )

        match = re.search(r"(\d+(?:[\.,]\d+)?)", sanitized_message)
        if not match:
            return None
        raw_value = match.group(1).replace(",", ".")
        try:
            return float(raw_value)
        except ValueError:
            return None

    @staticmethod
    def _evaluate_parameter_level(topic: str, measured_value: float) -> tuple[str, str]:
        value = float(measured_value)

        if topic in {"temperature", "st"}:
            if value < 18:
                return "scăzut", "poate produce senzație de frig și disconfort"
            if value <= 28:
                return "normal", "este în zona generală de confort"
            return "ridicat", "poate produce disconfort termic și oboseală"

        if topic in {"humidity", "sh"}:
            if value < 30:
                return "scăzut", "aerul poate deveni uscat pentru ochi și căi respiratorii"
            if value <= 70:
                return "normal", "este acceptabil pentru majoritatea spațiilor interioare"
            return "ridicat", "poate favoriza mucegaiul și senzația de aer greu"

        if topic == "pressure":
            if value < 980:
                return "scăzut", "poate indica schimbare de vreme și disconfort la persoane sensibile"
            if value <= 1030:
                return "normal", "este în interval atmosferic uzual"
            return "ridicat", "de regulă asociat cu vreme stabilă"

        if topic == "co2":
            if value < 400:
                return "scăzut", "de regulă semn de aer proaspăt abundent"
            if value <= 1000:
                return "normal", "ventilația este de obicei bună"
            if value <= 1500:
                return "moderat", "ventilația poate fi îmbunătățită"
            return "ridicat", "poate da somnolență, dureri de cap și scăderea concentrării"

        if topic == "lux":
            if value < 100:
                return "scăzut", "lumina poate fi insuficientă și obositoare pentru ochi"
            if value <= 500:
                return "normal", "este potrivit pentru majoritatea activităților de interior"
            return "ridicat", "poate crea disconfort vizual sau reflexii"

        if topic == "voc":
            if value < 250:
                return "scăzut", "nivelul este de regulă bun"
            if value <= 600:
                return "normal", "calitatea aerului este acceptabilă, dar merită monitorizată"
            return "ridicat", "poate provoca iritații și disconfort"

        if topic == "pm1":
            if value < 8:
                return "scăzut", "nivel favorabil pentru particule ultrafine"
            if value <= 25:
                return "normal", "nivel acceptabil"
            return "ridicat", "încărcare crescută de particule ultrafine"

        if topic == "pm25":
            if value < 10:
                return "scăzut", "calitate foarte bună pentru particule fine"
            if value <= 35:
                return "normal", "nivel acceptabil"
            if value <= 55:
                return "moderat", "sensibilii pot avea disconfort"
            return "ridicat", "risc crescut pentru confortul respirator"

        if topic == "pm10":
            if value < 20:
                return "scăzut", "nivel favorabil"
            if value <= 50:
                return "normal", "nivel acceptabil"
            if value <= 100:
                return "moderat", "poate irita căile respiratorii"
            return "ridicat", "încărcare mare de particule"

        return "necunoscut", "nu am un prag numeric stabil pentru acest indicator"

    @staticmethod
    def _build_parameter_brief(topic: str, measured_value: float | None = None) -> str:
        reference = AIR_QUALITY_REFERENCE.get(topic)
        if not reference:
            return "Nu am găsit informații pentru indicatorul cerut."

        name = reference.get("name", topic)
        unit = reference.get("unit", "unitate necunoscută")
        low = reference.get("low", "n/a")
        normal = reference.get("normal", "n/a")
        high = reference.get("high", "n/a")
        meaning = reference.get("meaning", "")
        effects = reference.get("effects", "")
        advice_low = reference.get("advice_low", "")
        advice_high = reference.get("advice_high", "")

        base = (
            f"{name}: {meaning}; unitatea este {unit}. "
            f"Orientativ: scăzut {low}, normal {normal}, ridicat {high}. "
            f"Ca efect, {effects}."
        )

        if measured_value is None:
            return (
                f"{base} Dacă este prea mic, recomandarea este: {advice_low}. "
                f"Dacă este prea mare, recomandarea este: {advice_high}."
            )

        level, impact = RuleBasedAirQualityChatbot._evaluate_parameter_level(topic, measured_value)
        measured_text = f"Valoarea {measured_value:g} {unit} este {level}"

        recommendation = advice_high
        if level == "scăzut":
            recommendation = advice_low
        elif level in {"normal", "moderat"}:
            recommendation = "continuă monitorizarea și ajustează ventilația/confortul în funcție de context"

        return f"{base} {measured_text}, deci {impact}; recomandare: {recommendation}."

    @staticmethod
    def _extract_concept_query(original_message: str, normalized_message: str) -> str | None:
        pattern = re.compile(r"(?:ce\s+inseamna|ce\s+înseamnă)\s+(.+)", re.IGNORECASE)
        match = pattern.search(original_message.strip())
        if match:
            return match.group(1).strip(" ?!.,")[:120]

        generic_concept_tokens = ["definitie", "definiție", "explica", "explică", "unitate", "prag", "interval"]
        if any(token in normalized_message for token in generic_concept_tokens):
            cleaned = re.sub(r"\s+", " ", original_message).strip(" ?!.,")
            return cleaned[:120]

        return None

    @staticmethod
    def _search_web_summary(query: str) -> str | None:
        endpoint = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        try:
            response = requests.get(endpoint, params=params, timeout=CHATBOT_WEB_SEARCH_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError, json.JSONDecodeError):
            return None

        abstract_text = (payload.get("AbstractText") or "").strip()
        if abstract_text:
            return abstract_text

        snippets: list[str] = []
        for topic in payload.get("RelatedTopics") or []:
            if isinstance(topic, dict):
                text = (topic.get("Text") or "").strip()
                if text:
                    snippets.append(text)

                for nested_topic in topic.get("Topics") or []:
                    if isinstance(nested_topic, dict):
                        nested_text = (nested_topic.get("Text") or "").strip()
                        if nested_text:
                            snippets.append(nested_text)

            if len(snippets) >= CHATBOT_WEB_SEARCH_MAX_SNIPPETS:
                break

        if not snippets:
            return RuleBasedAirQualityChatbot._search_wikipedia_summary(query)

        return " ".join(snippets[:CHATBOT_WEB_SEARCH_MAX_SNIPPETS])

    @staticmethod
    def _search_wikipedia_summary(query: str) -> str | None:
        encoded_query = requests.utils.quote(query)
        candidates = [
            f"https://ro.wikipedia.org/api/rest_v1/page/summary/{encoded_query}",
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}",
        ]

        for url in candidates:
            try:
                response = requests.get(url, timeout=CHATBOT_WEB_SEARCH_TIMEOUT_SECONDS)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError, json.JSONDecodeError):
                continue

            extract = (payload.get("extract") or "").strip()
            if extract:
                return extract

        return None


def _serialize_chat_context(context: ChatbotContext) -> dict[str, Any]:
    return {
        "latest_measurement": context.latest_measurement,
        "selected_device_identifier": context.selected_device_identifier,
        "available_devices": context.available_devices,
        "latest_prediction": (context.model_outputs or {}).get("latest_prediction"),
        "latest_training": (context.model_outputs or {}).get("latest_training"),
        "reference_ranges": AIR_QUALITY_REFERENCE,
        "learned_notes": [note.get("note") for note in (context.learned_notes or []) if note.get("note")],
    }


def _build_ollama_messages(
    message: str,
    context: ChatbotContext,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    context_payload = _serialize_chat_context(context)
    context_json = json.dumps(context_payload, ensure_ascii=False, default=str)

    system_prompt = (
        "Ești asistentul Air Quality AI. Răspunzi în română naturală, prietenoasă și clară. "
        "Răspunsul trebuie să aibă 2-5 propoziții, fără jargon inutil și fără ton robotic. "
        "Folosești doar datele din context. Contextul conține deja cea mai recentă predicție "
        "disponibilă (latest_prediction), obținută automat pentru tine — NU cere niciodată "
        "utilizatorului să ruleze Predict dacă latest_prediction există deja în context, "
        "ci răspunde direct folosind acele valori (inclusiv pentru anomalii, recomandări, PM2.5 sau CO2). "
        "Cere rularea Predict/Train doar dacă informația relevantă lipsește cu adevărat din context. "
        "Dacă utilizatorul cere o predicție generală, fără să numească un singur parametru "
        "(ex: 'vreau o predicție', 'cum e calitatea aerului'), rezumă TOATE valorile relevante din "
        "input_values/feature_assessment (temperatură, umiditate, PM2.5, PM10, CO2), nu doar una singură. "
        "Dacă utilizatorul întreabă despre un singur parametru (ex: doar PM2.5), poți răspunde punctual pe acela. "
        "Ține cont de istoricul conversației: dacă un mesaj anterior a menționat un parametru sau un orizont de timp "
        "(ex: 'peste 2 ore'), iar mesajul curent este un follow-up scurt (ex: 'dar peste 3 ore?'), presupune același "
        "parametru și răspunde coerent, fără să spui că nu înțelegi contextul. "
        "Pentru întrebări conceptuale (unități, praguri, intervale), folosești reference_ranges "
        "și NU răspunzi că lipsesc datele dacă informația există în reference_ranges. "
        "Contextul poate conține și learned_notes: instrucțiuni sau fapte pe care utilizatorul ți le-a spus anterior "
        "(ex: ce să verifici când un senzor e deconectat). Dacă mesajul curent se leagă de un learned_note, "
        "folosește-l ca sursă de adevăr și aplică-l direct, în loc să dai un răspuns generic sau să spui că nu știi. "
        "Închide răspunsul, când are sens, cu o întrebare scurtă de continuare."
    )

    few_shot_user_1 = "E grav PM2.5 acum?"
    few_shot_assistant_1 = (
        "PM2.5 pare ridicat în datele actuale, deci e bine să limitezi expunerea afară pentru moment. "
        "Dacă poți, aerisește scurt și folosește filtrare în interior. "
        "Vrei să verificăm și CO2 ca să vedem dacă ventilația e suficientă?"
    )

    few_shot_user_2 = "Cum stă modelul?"
    few_shot_assistant_2 = (
        "Pot să-ți spun rapid performanța modelului dacă am un training recent. "
        "Dacă nu există rezultate de antrenare, rulează Train și revin cu acuratețe, precizie și F1."
    )

    few_shot_user_2b = "Vreau o predicție"
    few_shot_assistant_2b = (
        "Din latest_prediction, calitatea aerului este momentan bună: temperatura și umiditatea sunt normale, "
        "iar PM2.5, PM10 și CO2 se încadrează în limite obișnuite. "
        "Vrei să detaliez vreunul dintre acești indicatori?"
    )

    few_shot_user_3 = "Ce inseamna ug/m3?"
    few_shot_assistant_3 = (
        "µg/m³ inseamna micrograme pe metru cub, adica masa de particule din aer. "
        "La PM2.5, orientativ: sub 35 e bun, 35-55 moderat, peste 55 ridicat. "
        "Vrei sa-ti explic si ppm pentru CO2?"
    )

    few_shot_user_4 = "Detectează anomalii"
    few_shot_assistant_4 = (
        "Am verificat valorile curente din latest_prediction și nu văd abateri majore față de intervalele normale. "
        "Toți indicatorii (temperatură, umiditate, PM2.5, CO2) se încadrează în limite obișnuite. "
        "Vrei să-ți arăt și recomandările pe baza acestor date?"
    )

    live_user_payload = (
        "Mesaj utilizator:\n"
        f"{message}\n\n"
        "Context structurat (JSON):\n"
        f"{context_json}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_user_1},
        {"role": "assistant", "content": few_shot_assistant_1},
        {"role": "user", "content": few_shot_user_2},
        {"role": "assistant", "content": few_shot_assistant_2},
        {"role": "user", "content": few_shot_user_2b},
        {"role": "assistant", "content": few_shot_assistant_2b},
        {"role": "user", "content": few_shot_user_3},
        {"role": "assistant", "content": few_shot_assistant_3},
        {"role": "user", "content": few_shot_user_4},
        {"role": "assistant", "content": few_shot_assistant_4},
    ]
    for history_message in (conversation_history or [])[-12:]:
        role = history_message.get("role")
        content = (history_message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": live_user_payload})
    return messages


def _generate_ollama_reply(
    message: str,
    context: ChatbotContext,
    conversation_history: list[dict[str, str]] | None = None,
) -> str | None:
    if not CHATBOT_USE_OLLAMA:
        return None

    payload = {
        "model": OLLAMA_MODEL,
        "messages": _build_ollama_messages(message, context, conversation_history),
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response_payload = response.json()
    except (
        requests.Timeout,
        requests.ConnectionError,
        requests.HTTPError,
        json.JSONDecodeError,
        ValueError,
    ):
        return None

    message_payload = response_payload.get("message") or {}
    reply = (message_payload.get("content") or "").strip()
    if not reply:
        return None

    return reply


def get_chatbot_reply(
    message: str,
    model_outputs: dict[str, Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    device_identifier: str | None = None,
) -> str:
    chatbot = RuleBasedAirQualityChatbot()

    normalized = (message or "").strip().lower()

    if CHATBOT_ENABLE_LEARNING and chatbot.is_teaching_message(normalized):
        note_text = chatbot.extract_taught_note(message, normalized)
        if save_chatbot_note(note_text, device_identifier=device_identifier):
            return (
                f"Am reținut: \"{note_text}\". "
                "Data viitoare când apare o situație similară, o să țin cont de asta automat."
            )

    is_live_question = chatbot.is_live_data_question(normalized)
    is_location_question = chatbot.is_location_question(normalized)
    is_devices_question = chatbot.is_devices_question(normalized)
    is_capabilities_question = chatbot.is_capabilities_question(normalized)
    # A forecast question (e.g. "cat va fi temperatura peste 2 ore") mentions a parameter name too,
    # so it must win over the generic reference/definition lookup below instead of being intercepted.
    is_forecast_question = chatbot.is_forecast_question(normalized)
    if is_live_question or chatbot.needs_live_prediction(normalized):
        model_outputs = model_outputs if model_outputs is not None else {}
        chatbot.try_build_live_prediction(model_outputs, device_identifier=device_identifier)
    detected_topics = chatbot.detect_reference_topics(normalized)

    if detected_topics and not is_live_question and not is_forecast_question:
        reference_reply = chatbot.build_reference_reply(message, normalized)
        if reference_reply:
            return reference_reply

    reference_intent_tokens = [
        "ug/m3",
        "µg/m3",
        "µg/m³",
        "ppm",
        "ce inseamna",
        "ce înseamnă",
        "unitate",
        "prag",
        "interval",
        "catitate",
        "cantitate",
        "moderat",
        "ridicat",
        "bun",
    ]
    if any(token in normalized for token in reference_intent_tokens) and not is_live_question and not is_forecast_question:
        reference_reply = chatbot.build_reference_reply(message, normalized)
        if reference_reply:
            return reference_reply

    context = chatbot.build_context(
        model_outputs=model_outputs,
        device_identifier=device_identifier,
    )
    if CHATBOT_ENABLE_LEARNING:
        context.learned_notes = get_chatbot_notes(device_identifier=device_identifier)

    if (
        is_live_question
        or is_location_question
        or is_forecast_question
        or is_devices_question
        or is_capabilities_question
    ):
        return chatbot.generate_reply(message=message, context=context, conversation_history=conversation_history)

    llm_reply = _generate_ollama_reply(
        message=message,
        context=context,
        conversation_history=conversation_history,
    )
    if llm_reply:
        return llm_reply

    return chatbot.generate_reply(message=message, context=context, conversation_history=conversation_history)


def get_chatbot_reply_details(
    message: str,
    model_outputs: dict[str, Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    device_identifier: str | None = None,
) -> dict[str, Any]:
    """Return the chatbot reply together with BERT routing evidence."""
    bert_feature = None
    bert_available = False
    try:
        bert_features = detect_sensor_features(message)
        bert_feature = bert_features[0] if bert_features else None
        bert_available = True
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        bert_available = False

    reply = get_chatbot_reply(
        message,
        model_outputs=model_outputs,
        conversation_history=conversation_history,
        device_identifier=device_identifier,
    )
    return {
        "reply": reply,
        "models_used": ["rule_based_chatbot", "bert_sensor_intent"] if bert_available else ["rule_based_chatbot"],
        "bert_available": bert_available,
        "bert_model": get_bert_model_name(),
        "bert_feature": bert_feature,
        "response_engine": "ollama" if CHATBOT_USE_OLLAMA else "rule_based_chatbot",
    }


def get_chatbot_welcome_message() -> str:
    return WELCOME_MESSAGE
