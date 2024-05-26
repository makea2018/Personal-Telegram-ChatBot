from chatbots.saiga_chatbot import SaigaBot
import json

# Загрузка истории чата
chat_history = [
    {"role": "system", "content": "Ты - умный чат бот, который отвечает максимально четко и по делу на вопросы пользователя. Ответы ты даешь на том языке, на котором с тобой общается пользователь."},
]

# Инициализация чат-бота
bot = SaigaBot("weights/Saiga-Llama-3-8B", chat_history)

# Вопрос боту
answer = bot.generate_answer("Привет, расскажи о себе. Чем ты отличаешь от обычной Llama-3-8B от компании Meta?")

# Сохранение истории чата
bot.save_chat_history()

print(answer)
