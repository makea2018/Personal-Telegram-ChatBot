from chatbots.kandinsky_bot import ImageGenerator


# Пути до весов моделей Kandinsky 2.2
prior_path = "D:/Hugging_face_models/Diffusion models/kandinsky-2-2-prior"
decoder_path = "D:/Hugging_face_models/Diffusion models/kandinsky-2-2-decoder"

# Объект Генератор Картинок
KandinskyBot = ImageGenerator(prior_path, decoder_path)

# Генерация изображения
prompt = "anime drawing, expression serious, close-up intensity, masterpiece, waist up, best quality, ultra-detailed, cinematic beautiful lighting, intricate details, looking at viewer, depth of field--ar 2:3 --s 200 --niji 5"
neg_prompt = "tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, poorly drawn ears Sampler: DPM++ 2M Karras"
image = KandinskyBot.generate_image(prompt, neg_prompt)

# Вывод изображения на экран
KandinskyBot.show_image()
