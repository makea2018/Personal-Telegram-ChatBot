import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")

torch.random.manual_seed(0)
model_id = "weights/models--NousResearch--Meta-Llama-3-8B-Instruct"

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto"
)

# Проверка что есть доступ к GPU
assert torch.cuda.is_available(), "Модель LLama 3 работает только на GPU ..."
device = torch.cuda.current_device()

# Перенос модели на GPU
# model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)
newline_token = tokenizer.encode("\n")[0]


# Задаем сообщения для диалога-беседы
messages = [
    {"role": "assistant", "content": "Ты - талантливый ML-инженер, который специализируется на задачах компьютьерного зрения. Также ты прекрасно понимаешь русский язык и даешь ответы исключительно на русском языке, если тебя не попросят дать ответ на другом языке, например, английском."}
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    eos_token_id=newline_token,
    pad_token_id=tokenizer.eos_token_id
)

generation_args = {
    "max_new_tokens": 200,
    "return_full_text": False,
    "temperature": 0.2,
    "do_sample": False,
    "num_return_sequences": 1
}

# Реализация диалога с LLama 3-8B
print("Assistant: Привет! Я Llama 3. Создана компанией Facebook. Я готова ответить на твои вопросы. Для завершения разговора напиши 'Пока'.")

while True:
    user_input = input("User: ")

    message = {"role": "user", "content": user_input}

    # Добавляем в историю беседы вопрос пользователя
    messages.append(message)

    if user_input.lower() == "пока":
        print("Assistant: До свидания! Спасибо за общение.")
        break

    # Ответ модели
    answer = pipe(messages, **generation_args)

    print("Assistant: " + answer[0]['generated_text'])

    # Добавляем в историю беседы ответ чат-бота
    messages.append({"role": "assistant", "content": answer[0]['generated_text']})
