from transformers import pipeline
from huggingface_hub import snapshot_download

# Модель №1 - Llama-3-8B-Instruct
# model_1 = snapshot_download("NousResearch/Meta-Llama-3-8B-Instruct", cache_dir="weights")
# Модель №1 - Saiga-Llama-3-8B-Instruct (от IlyaGusev)
model_2 = snapshot_download("IlyaGusev/saiga_llama3_8b", cache_dir="weights")
