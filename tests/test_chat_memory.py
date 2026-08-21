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


def test_chatbot_answers_device_location_from_latest_measurement(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame([
            {
                "device_identifier": "dev-1",
                "location": "Laborator 2",
                "latitude": 44.4321,
                "longitude": 26.1039,
            }
        ]),
    )

    reply = get_chatbot_reply(
        "Unde se afla dispozitivul?",
        device_identifier="dev-1",
    )

    assert "Laborator 2" in reply
    assert "44.43210" in reply
    assert "26.10390" in reply


def test_chatbot_reports_missing_location_for_selected_device(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame(),
    )

    reply = get_chatbot_reply("Unde este senzorul?", device_identifier="dev-x")

    assert "dev-x" in reply
    assert "nu pot determina locația" in reply.lower()


def test_chatbot_uses_location_table_when_measurement_has_no_location(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame([{"device_identifier": "dev-77", "temperature": 22.0}]),
    )
    monkeypatch.setattr(
        "app.services.chatbot.get_device_location_details",
        lambda device_identifier: {
            "location": "Depozit Nord",
            "latitude": 46.7712,
            "longitude": 23.6236,
            "source_table": "location",
            "source_id_column": "device_identifier",
        },
    )

    reply = get_chatbot_reply("Unde se afla dispozitivul?", device_identifier="dev-77")

    assert "Depozit Nord" in reply
    assert "46.77120" in reply
    assert "23.62360" in reply


def test_chatbot_lists_all_devices_from_account(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame([{"device_identifier": "dev-1"}]),
    )
    monkeypatch.setattr(
        "app.services.chatbot.get_devices_with_location",
        lambda: [
            {"device_identifier": "dev-1", "location_label": "Lab A"},
            {"device_identifier": "dev-2", "location_label": "Birou"},
        ],
    )

    reply = get_chatbot_reply("Ce dispozitive sunt pe cont?")

    assert "2 dispozitive" in reply
    assert "dev-1" in reply
    assert "Lab A" in reply
    assert "dev-2" in reply


def test_chatbot_reports_capabilities_and_device_count(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "app.services.chatbot.get_measurements",
        lambda **kwargs: pd.DataFrame([{"device_identifier": "dev-1"}]),
    )
    monkeypatch.setattr(
        "app.services.chatbot.get_devices_with_location",
        lambda: [{"device_identifier": "dev-1", "location_label": "Lab A"}],
    )

    reply = get_chatbot_reply("Ce poți face, toate funcțiile?")

    assert "funcțiile principale" in reply or "funcțiile" in reply
    assert "1 dispozitive" in reply
