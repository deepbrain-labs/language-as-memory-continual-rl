import torch
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel

def train_rlhf(args):
    # Config
    config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_ppo_epochs=4,
        seed=42,
        use_cpu=not torch.cuda.is_available(),
        output_dir=args.output_dir
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    # Dataset
    print(f"Loading data from {args.train_file}")
    def build_dataset(data_path):
        ds = load_dataset("json", data_files=data_path, split="train")
        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["prompt"])
            sample["query"] = sample["input_ids"]
            return sample
        ds = ds.map(tokenize, batched=False)
        ds = ds.remove_columns(["prompt", "chosen", "rejected"])
        return ds

    dataset = build_dataset(args.train_file)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        print(f"Limiting dataset to {args.max_train_samples} samples")
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))
    
    # Model (Policy)
    print("Loading Policy Model...")
    bnb_config = None
    if args.load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    model.config.use_cache = False 
    
    if args.dpo_model_dir:
        print(f"Loading DPO adapter from {args.dpo_model_dir}")
        try:
            model = PeftModel.from_pretrained(model, args.dpo_model_dir)
            print("Adapter loaded.")
        except Exception as e:
            print(f"Failed to load adapter: {e}")

    # Reward Model
    print(f"Loading Reward Model from {args.reward_model_path}")
    # Load Base
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load Adapter if path is a directory (LoRA)
    if args.reward_model_path and os.path.exists(os.path.join(args.reward_model_path, "adapter_config.json")):
        print(f"Loading RM Adapter from {args.reward_model_path}")
        reward_model = PeftModel.from_pretrained(reward_model, args.reward_model_path)
    
    # Value Model (Initialized from RM base)
    print("Loading Value Model...")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    if value_model.config.pad_token_id is None:
        value_model.config.pad_token_id = tokenizer.pad_token_id
     
    # Load Value Adapter if RM was LoRA
    if args.reward_model_path and os.path.exists(os.path.join(args.reward_model_path, "adapter_config.json")):
        print(f"Loading VM Adapter from {args.reward_model_path}")
        value_model = PeftModel.from_pretrained(value_model, args.reward_model_path)

    print("Initializing PPOTrainer...")
    def collator(data):
        return tokenizer.pad(data, padding=True, return_tensors="pt")

    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=model,
        ref_model=None, 
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting PPO Training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dpo_model_dir", type=str, default="results_dpo")
    parser.add_argument("--reward_model_path", type=str, default="results_rm")
    parser.add_argument("--train_file", type=str, default="data/dpo_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="results_rlhf")
    parser.add_argument("--lr", type=float, default=1.41e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0, help="Set to 0 for Windows")
    
    args = parser.parse_args()

    if os.name == 'nt' and args.num_workers > 0:
        print(f"Warning: forcing num_workers=0 (was {args.num_workers}) on Windows to prevent DataLoader hang.")
        args.num_workers = 0

    train_rlhf(args)
