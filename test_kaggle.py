import os

# Definisci l'ID del dataset
dataset_id = "mdismielhossenabir/sentiment-analysis"

print(os.getenv('secrets.KAGGLE_API_TOKEN'))
import os
os.environ['KAGGLE_CONFIG_DIR'] = "/workspaces/Progetto-Monitoraggio-della-reputazione-online-di-un-azienda"

import kaggle

# Scarica il dataset utilizzando l'API di Kaggle
kaggle.api.dataset_download_files(dataset_id, path="./data", unzip=True)