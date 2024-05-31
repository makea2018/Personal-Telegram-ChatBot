from typing import Optional, Dict
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
import matplotlib.pyplot as plt


# Конфиги для генерации изображения
generate_configs = {
    "height": 800,
    "width": 1024,
    "num_inference_steps": 50,
    "num_images_per_prompt" : 1,
    "guidance_scale" : 1.0,
}

prior_path = "D:/Hugging_face_models/Diffusion models/kandinsky-2-2-prior"
decoder_path = "D:/Hugging_face_models/Diffusion models/kandinsky-2-2-decoder"

class ImageGenerator:
    def __init__(self, prior_path: str, decoder_path: str, generate_configs: Dict):
        # Инициализация сгенерированного изображения
        self.image = None

        # Инициализация пайплайнов необходимых для запуска Kandinsky 2.2
        self.pipe_prior = KandinskyV22PriorPipeline.from_pretrained(prior_path)
        self.pipe = KandinskyV22Pipeline.from_pretrained(decoder_path)

        # Загрузка весов моделей на видеокарту
        if torch.cuda.is_available():
            self.pipe_prior.to("cuda")
            self.pipe.to("cuda")

        # Инициализация конфигов генерации модели
        self.generate_configs = generate_configs

    # Метод для генерации картинки
    def generate_image(self, prompt: str, negative_prompt: Optional[str] = None):
        # Кодирование текста промтов
        prompt, negative_prompt = self.pipe_prior([prompt, negative_prompt]).to_tuple()

        # Генерация изображения
        self.image = self.pipe(prompt_embeds=prompt, negative_prompt_embeds=negative_prompt,
                          **generate_configs)[0]

        return self.image

    # Метод для сохранения сгенерированного изображения на локальный диск
    def save_image(self, fname: str):
        if self.image is None:
            print("Сначала нужно сгенерировать изображение, затем будет возможность его сохранить!")
        else:
            self.image.save(fname)

    def show_image(self):
        if self.image is None:
            print("Сначала нужно сгенерировать изображение, затем будет возможность его отобразить!")
        else:
            # Вывод сгенерированного изображения
            plt.imshow(self.image)
            plt.show()
