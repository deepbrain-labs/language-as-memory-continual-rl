import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def get_vram_usage():
    return torch.cuda.memory_allocated() / (1024**3)

def load_phi2_test():
    model_name = "microsoft/phi-2"
    
    print(f"Initial VRAM: {get_vram_usage():.2f} GB")
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"VRAM after load: {get_vram_usage():.2f} GB")
    
    print("Preparing for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["Wqkv", "fc1", "fc2"], # Phi-2 specific
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print(f"VRAM after LoRA: {get_vram_usage():.2f} GB")
    
    assert get_vram_usage() < 8.0, "VRAM usage exceeded 8GB!"
    
    print("\nRunning dummy forward pass...")
    inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("✔ Forward pass successful.")
    print(f"Final VRAM: {get_vram_usage():.2f} GB")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Skipping test: GPU not available.")
    else:
        try:
            load_phi2_test()
            print("\n✅ PHASE 0: Model Loading Test Passed.")
        except AssertionError as e:
            print(f"\n❌ PHASE 0: VRAM Constraint Failed: {e}")
        except Exception as e:
            print(f"\n❌ PHASE 0: Model Loading Failed: {e}")
