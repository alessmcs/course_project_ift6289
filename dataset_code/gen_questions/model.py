from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
#     local_dir="/data/rech/jaouaime/Llama-3.3-70B-Instruct-GPTQ-Int4",
#     local_dir_use_symlinks=False,
#     resume_download=True
# )

snapshot_download(
    repo_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    local_dir="/Tmp/mancasat/models/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
    local_dir_use_symlinks=False,
    resume_download=True
)