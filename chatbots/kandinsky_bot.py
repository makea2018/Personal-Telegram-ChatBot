from typing import Optional, Dict
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
import matplotlib.pyplot as plt


# Конфиги для генерации изображения
generate_configs = {
    "height": 800,
    "width": 1024,
    "num_inference_steps": 50,
    "num_images_per_prompt": 1,
    "guidance_scale": 1.0,
}


class ImageGenerator:
    def __init__(self, prior_path: str, decoder_path: str, generate_configs: Dict = generate_configs):
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
        if negative_prompt is None:
            prompt, negative_prompt = self.pipe_prior(prompt).to_tuple()
        else:
            prompt, negative_prompt = self.pipe_prior(prompt, negative_prompt).to_tuple()

        # Генерация изображения
        self.image = self.pipe(image_embeds=prompt, negative_image_embeds=negative_prompt,
                               **generate_configs).images[0]

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
