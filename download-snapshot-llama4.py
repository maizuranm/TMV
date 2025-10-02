from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    local_dir="/home/zura/models/llama4_maverick_bf16" #input the path directory where the model should be downloaded
)

snapshot_download(
    repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    local_dir="/home/zura/models/llama4_maverick_fp8" #input the path directory where the model should be downloaded)
