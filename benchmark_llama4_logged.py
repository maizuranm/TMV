import os, time, csv, argparse, psutil, torch #torch untuk GPU, timing, tensor. psutil untuk baca penggunaan RAM CPU. time utk kira masa(load,generate,latency). csv untuk simpan hasil benchmark dalam fail csv. angparse bagi kita run script dengan argumen CLI supaya lbh flexible
from transformers import AutoTokenizer, AutoModelForCausalLM #untuk load model/tokenizer dari Hugging Face

#Utility untuk detect GPU
def human_gpu():
    if not torch.cuda.is_available():
        return "CPU-only" #kalau tak ada gpu, maka result adalah "CPU only"
    name = torch.cuda.get_device_name(0)
    num = torch.cuda.device_count()
    return f"{num}x {name}" #kalau ada GPU, result adalah dlm string "1x H100-80GB"

#Utility untuk print memory usage
def print_memory(label=""):
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU] {label}  Alloc: {alloc:.2f} GB | Reserv: {reserv:.2f} GB") #kalau ada GPU tunjukkan allocated (yang digunakan oleh model sekarang) dammn yg reserved (torch simpan sbg buffer)
    print(f"[CPU] {label}  RAM: {ram:.2f} GB") #kalau takder GPU tunjukkan penggunaan RAM CPU

#Utility untuk write row ke dalam CSV
def write_csv(row, csv_path):
    header = [
        "run_id","model_path","precision","gpu_type_qty",
        "vram_peak_gb","context_tokens","max_new_tokens","batch",
        "tok_per_sec","latency_first_s","latency_ms_per_token","load_time_s","notes"
    ]
    write_header = not os.path.exists(csv_path) #Generate file dengan nama benchmark_results.csv. Klu file tak wujud tulis header. Kalau file dah wujud hanya append new row    
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row)

def benchmark(model_path, run_id, precision, gpu_label, prompt, max_new_tokens=256, batch=1, csv_path="benchmark_results.csv", notes=""):
    print(f"\n===== Benchmarking {model_path} | {precision} | batch={batch} | max_new={max_new_tokens} =====") 

    # Load
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # NOTE: Untuk FP8 checkpoint, biar loader default—torch_dtype arg tak akan “menukar” FP8 file.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,   # BF16 runtime; FP8 weights tetap akan dihormati oleh loader
        device_map="auto"
    )
    load_time = time.time() - t0

    # Input (support batch)
    texts = [prompt] * batch
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    context_len = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print_memory("after load")

    # First-token latency: generate 1 token dahulu
    t_first0 = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_first = time.time() - t_first0

    # Full generation
    t_gen0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_gen = time.time() - t_gen0

    # Metrics
    gen_tokens_each = out.shape[1] - inputs["input_ids"].shape[1]
    total_gen_tokens = gen_tokens_each * batch
    tok_per_sec = total_gen_tokens / t_gen if t_gen > 0 else float("inf")
    ms_per_token = (t_gen / total_gen_tokens * 1000) if total_gen_tokens > 0 else 0.0

    # VRAM peak (lebih tepat dari allocated/reserved current)
    if torch.cuda.is_available():
        vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    else:
        vram_peak_gb = 0.0

    print_memory("after gen")
    print(f"First token latency: {t_first:.3f}s")
    print(f"Generated {total_gen_tokens} tokens in {t_gen:.2f}s  → {tok_per_sec:.2f} tok/s")
    print(f"Avg latency: {ms_per_token:.2f} ms/token")
    print(f"Load time: {load_time:.2f}s")
    print("Sample output:\n", tokenizer.decode(out[0], skip_special_tokens=True)[:500], "...\n")

    row = {
        "run_id": run_id,
        "model_path": model_path,
        "precision": precision,
        "gpu_type_qty": gpu_label or human_gpu(),
        "vram_peak_gb": f"{vram_peak_gb:.2f}",
        "context_tokens": context_len,
        "max_new_tokens": max_new_tokens,
        "batch": batch,
        "tok_per_sec": f"{tok_per_sec:.2f}",
        "latency_first_s": f"{t_first:.3f}",
        "latency_ms_per_token": f"{ms_per_token:.2f}",
        "load_time_s": f"{load_time:.2f}",
        "notes": notes or ""
    }
    write_csv(row, csv_path)
    print("Logged to:", csv_path)
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--precision", type=str, default="BF16/FP8-ckpt")
    ap.add_argument("--gpu_label", type=str, default="")
    ap.add_argument("--prompt", type=str, default="Summarize the importance of AI in nuclear security in 3 bullet points. Explain in simple English suitable for policymakers.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--csv_path", type=str, default="benchmark_results.csv")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    benchmark(
        model_path=args.model_path,
        run_id=args.run_id,
        precision=args.precision,
        gpu_label=args.gpu_label,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch=args.batch,
        csv_path=args.csv_path,
        notes=args.notes
    )
