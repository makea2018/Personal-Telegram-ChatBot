from transformers import pipeline
from huggingface_hub import snapshot_download

model = snapshot_download("microsoft/Phi-3-small-8k-instruct", cache_dir="weights")
