from huggingface_hub import snapshot_download

# Загрузка модели SD-XL 1.0-base
midjourney_v6 = snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", cache_dir="weights")
