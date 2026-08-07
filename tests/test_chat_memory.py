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
