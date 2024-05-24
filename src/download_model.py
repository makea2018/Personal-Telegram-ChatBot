from transformers import pipeline
from huggingface_hub import snapshot_download

model = snapshot_download("NousResearch/Meta-Llama-3-8B-Instruct", cache_dir="weights")
