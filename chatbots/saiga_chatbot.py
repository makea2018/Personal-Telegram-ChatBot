from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import warnings
warnings.filterwarnings("ignore")

# Конфиги для генерации модели
generate_configs = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 0.2,
    "top_p": 0.1,
}


class SaigaBot():
    def __init__(self, model_path, chat_history, generate_configs=generate_configs):
        # Инициализация истории чата
        self.chat_history = chat_history

        # Инициализация конфигов генерации модели
        self.generate_configs = generate_configs

        # Инициализация токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Инициализация модели
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Инициализация eos_token
        self.terminators = [self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # Метод для генерации ответа
    def generate_answer(self, message):
        # Добавление в историю чата сообщение
        message = {"role": "user", "content": message}
        self.chat_history.append(message)

        # Токенизация предложений
        input_ids = self.tokenizer.apply_chat_template(
            self.chat_history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Генерация моделью ответа и преобразование ids в текстовое представление
        outputs = self.model.generate(
            input_ids,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.generate_configs
            )

        # Формирование ответа в виде str
        response = outputs[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(response, skip_special_tokens=True)

        # Обновление истории чата
        self.chat_history.append({"role": "system", "content": answer})

        return answer

    # Метод для удаления истории чата
    def clear_chat_history(self):
        # Удаление всей беседы, кроме промта генерации модели
        self.chat_history = self.chat_history[0]

    # Метод для сохранения истории чата
    def save_chat_history(self):
        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=4)
