from app.services.chatbot import ChatbotContext, _build_ollama_messages, get_chatbot_reply


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
