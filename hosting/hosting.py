import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SPACE_REPO = os.getenv("HF_SPACE_REPO", "nalamrc/tourism-wellness-app")
BASE_DIR = Path(os.getenv("GITHUB_SRC_ART_BASE_DIR", ".")).expanduser()
DEPLOY_DIR = BASE_DIR / "deployment"

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_SPACE_REPO, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

for filename in ["Dockerfile", "app.py", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=str(DEPLOY_DIR / filename),
        path_in_repo=filename,
        repo_id=HF_SPACE_REPO,
        repo_type="space",
    )

{"status": "Upload complete when valid HF_TOKEN and HF_SPACE_REPO are provided."}
