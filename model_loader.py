from config import ModelConfig

import os
from huggingface_hub import snapshot_download


model_name, model_path = ModelConfig.model_name, ModelConfig.model_path

if not os.path.exists(model_path):
    os.makedirs(model_path)


print(f"⤵️  Загрузка модели {model_name} ...")

snapshot_download(
    repo_id=model_name,
    local_dir=model_path
)

print(f"✅ Модель сохранена в {model_path}")
