import argparse
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType
from trl import RewardTrainer, RewardConfig

def train_reward_model(args):
    print(f"Loading data from {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")

    # Quantization
    bnb_config = None
    if args.load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # Model
    print(f"Loading Reward Model base: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA for RM
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
        modules_to_save=["score"]
    )

    training_args = RewardConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    print("Initializing RewardTrainer...")
    try:
        trainer = RewardTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
        )
    except TypeError:
        # Fallback for older TRL versions that use 'tokenizer'
        print("Fallback to 'tokenizer' argument for RewardTrainer")
        trainer = RewardTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
        )

    print("Starting RM Training...")
    trainer.train()
    
    print(f"Saving RM to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--train_file", type=str, default="data/dpo_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="results_rm")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0, help="Set to 0 for Windows to avoid DataLoader hangs")

    args = parser.parse_args()

    if os.name == 'nt' and args.num_workers > 0:
        print(f"Warning: forcing num_workers=0 (was {args.num_workers}) on Windows to prevent DataLoader hang.")
        args.num_workers = 0

    train_reward_model(args)
