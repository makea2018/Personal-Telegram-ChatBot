from time import time
from chatbots.saiga_chatbot import SaigaBot

# Инициализация чат-бота
bot = SaigaBot("weights/Saiga-Llama-3-8B")

# Вопрос боту
start = time()
answer = bot.generate_answer("Кто играл роль капитана Джека Воробья в Пиратах Карибского моря?")
end = time() - start
print("Ответ обычный: " + answer)

print(f"Время: {end:.2f} сек")
