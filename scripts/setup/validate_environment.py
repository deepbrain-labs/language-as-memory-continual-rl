import sys
import os
import torch
import psutil

def check_gpu():
    print("=== GPU Check ===")
    if not torch.cuda.is_available():
        print("❌ CUDA NOT available. Please enable GPU in Colab runtime.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✔ GPU Detected: {gpu_name}")
    
    # Check for T4 or better
    if "T4" in gpu_name or "A100" in gpu_name or "V100" in gpu_name or "L4" in gpu_name:
        print("✔ Compatible GPU found.")
    else:
        print(f"⚠ Warning: {gpu_name} might be slow for training.")

    # Check VRAM
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    print(f"✔ VRAM: {vram_gb:.2f} GB")
    
    if vram_gb < 14:
        print("⚠ Warning: Less than 14GB VRAM. Phi-2 training might be tight.")
    
    return True

def check_dependencies():
    print("\n=== Dependency Check ===")
    try:
        import transformers
        import bitsandbytes
        import accelerate
        import peft
        import wandb
        print(f"✔ transformers: {transformers.__version__}")
        print(f"✔ accelerate: {accelerate.__version__}")
        print(f"✔ bitsandbytes: {bitsandbytes.__version__}")
        print(f"✔ peft: {peft.__version__}")
        print(f"✔ wandb: {wandb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_storage():
    print("\n=== Storage Check ===")
    # Just check if we can write to the current directory or specific drive path
    try:
        test_file = "test_write.tmp"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("✔ Write access confirmed.")
        
        # Check disk space
        hdd = psutil.disk_usage('.')
        free_gb = hdd.free / (1024**3)
        print(f"✔ Free Disk Space: {free_gb:.2f} GB")
        
        if free_gb < 10:
             print("⚠ Warning: Low disk space (<10GB).")
             
    except Exception as e:
        print(f"❌ Write check failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Starting Phase 0 Environment Check...\n")
    gpu_ok = check_gpu()
    deps_ok = check_dependencies()
    storage_ok = check_storage()
    
    if gpu_ok and deps_ok and storage_ok:
        print("\n✅ PHASE 0: Environment Ready.")
    else:
        print("\n❌ PHASE 0: Environment Issues Found.")
        sys.exit(1)
