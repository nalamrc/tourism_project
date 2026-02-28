import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SPACE_REPO = os.getenv("HF_SPACE_REPO", "nalamrc/tourism-wellness-app")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_SPACE_REPO, repo_type="space", space_sdk="streamlit", private=False, exist_ok=True)

for filename in ["Dockerfile", "app.py", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/deployment/{filename}",
        path_in_repo=filename,
        repo_id=HF_SPACE_REPO,
        repo_type="space",
    )

{"status": "Upload complete when valid HF_TOKEN and HF_SPACE_REPO are provided."}
