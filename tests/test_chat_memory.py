from app.services.chatbot import ChatbotContext, _build_ollama_messages, get_chatbot_reply


def test_chatbot_reads_current_temperature_and_database_label(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame([{"temperature": 23.4, "quality_label": "good"}]),
    )
    monkeypatch.setattr(
        "app.services.chatbot.predict_air_quality",
        lambda: ("good", 0.91, {"temperature": 23.4}, {}),
    )

    reply = get_chatbot_reply("Ce temperatură este acum în camera mea și cum e eticheta?")

    assert "23.4" in reply
    assert "bună" in reply
    assert "baza de date" in reply


def test_chatbot_reports_missing_current_measurement(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame(),
    )

    reply = get_chatbot_reply("Care este temperatura actuală în cameră?")

    assert "Nu am găsit" in reply


def test_chatbot_reads_requested_dust_and_air_values_from_database(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame([{
            "temperature": 23.4,
            "humidity": 48.0,
            "pm25": 12.6,
            "pm10": 31.2,
            "co2": 742.0,
        }]),
    )

    reply = get_chatbot_reply("Care este valoarea prafului și a CO2 acum?")

    assert "PM10 (praf)=31.2 µg/m³" in reply
    assert "CO2=742.0 ppm" in reply
    assert "PM2.5" not in reply


def test_chatbot_uses_bert_sensor_intent_for_paraphrased_question(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame([{"pm10": 31.2}]),
    )
    monkeypatch.setattr(
        "app.services.chatbot.detect_sensor_features",
        lambda message: ["pm10"],
    )

    reply = get_chatbot_reply("Cât de încărcat este aerul cu particule acum?")

    assert "PM10 (praf)=31.2 µg/m³" in reply


def test_ollama_messages_include_recent_chat_history():
    history = [
        {"role": "user", "content": f"Mesaj utilizator {index}"}
        for index in range(14)
    ]

    messages = _build_ollama_messages(
        message="Ce ai retinut?",
        context=ChatbotContext(),
        conversation_history=history,
    )

    history_contents = [message["content"] for message in messages[-13:-1]]

    assert history_contents == [f"Mesaj utilizator {index}" for index in range(2, 14)]
    assert messages[-1]["role"] == "user"
    assert "Ce ai retinut?" in messages[-1]["content"]


def test_chatbot_explains_generic_pm_term():
    reply = get_chatbot_reply("Ce masoara PM?")

    assert "PM2.5" in reply
    assert "PM10" in reply


def test_chatbot_identifies_pm10_as_dust_indicator():
    reply = get_chatbot_reply("Care indicator arata praful?")

    assert "PM10" in reply
    assert "praf" in reply.lower()


def test_chatbot_extracts_only_pm10_from_pm_10_question():
    reply = get_chatbot_reply("Care e diferenta dintre pm 1 si pm 10?")

    assert "PM1" in reply
    assert "PM10" in reply
    assert "PM2.5" not in reply
