from huggingface_hub import snapshot_download, hf_hub_download

# Загрузка модели SD-XL 1.0-base
# sd_xl_1_0_base = snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", cache_dir="weights")

# Загрузка модели kandinsky-2-2-prior
kandinsky_2_2_prior = snapshot_download("kandinsky-community/kandinsky-2-2-prior",
                                        cache_dir="weights")
