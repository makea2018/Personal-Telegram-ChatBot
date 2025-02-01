from chatbots.saiga_chatbot import SaigaBot
import json

# Загрузка истории чата
with open('chat_history.json', 'r', encoding='utf-8') as f:
    chat_history = json.load(f)

# Инициализация чат-бота
bot = SaigaBot("weights/Saiga-Llama-3-8B")

# Вопрос боту
answer = bot.generate_answer("Кто играл роль капитана Джека Воробья в Пиратах Карибского моря?")

print("Ответ обычный: " + answer)

# Меняем роль бота
bot.change_role("Ты - бывалый моряк, в своих ответах ты используешь морские слова: палундра, якорь мне в бухту и другие в этом роде.")

# Вопрос боту
answer = bot.generate_answer("Кто играл роль капитана Джека Воробья в Пиратах Карибского моря?")

# Сохранение истории чата
# bot.save_chat_history()

print("Ответ в роли моряка: " + answer)
