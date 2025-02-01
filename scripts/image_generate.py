from diffusers import AutoPipelineForText2Image
import torch
import matplotlib.pyplot as plt
from time import time

pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

start = time()

prompt = "Swimming dolphin in the pond, wonderful style, warm pallete"
negative_prompt = "low quality, asymmetry"

image = pipe(prompt=prompt, negative_prompt=None, prior_guidance_scale =1.0, height=800, width=1024).images[0]
image.save("generated_pictures/delphin.jpeg")

final_time = time() - start
# Вывод сколько заняла генерация картинки
print(f"Время: {final_time:.2f} сек.")

# Вывод сгенерированного изображения
plt.imshow(image)
plt.show()
