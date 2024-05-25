from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

model_id = "weights/Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# История диалога с моделью
messages = [
    {"role": "system", "content": "Ты - умный чат бот, который отвечает максимально четко и по делу на вопросы пользователя. Ответы ты даешь на том языке, на котором с тобой общается пользователь."},
    {"role": "user", "content": "Привет! Расскажи о себе."},
]

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Код диалога с моделью
while True:
    user_input = input("User: ")

    message = {"role": "user", "content": user_input}

    if user_input.lower() == "пока":
        print("Bot: До свидания! Спасибо за общение.")
        break

    # Добавляем в историю беседы вопрос пользователя
    messages.append(message)

    # Ответ модели
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=0.2,
        top_p=0.1
        )

    # Ответ
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)

    print("Bot: " + answer)

    # Добавляем в историю беседы ответ чат-бота
    messages.append({"role": "system", "content": answer})
