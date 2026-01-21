import argparse
import torch
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.llm.planner import Phi2Planner
from tqdm import tqdm

def evaluate_alignment(args):
    print("=== Alignment Evaluation ===")
    
    # Load Model (Base + Adapter)
    planner = Phi2Planner(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_lora=True,
        adapter_path=args.adapter_path,
        prompt_mode="constrained" # Use same mode as training
    )
    
    # Load Test Data
    print(f"Loading test data: {args.test_file}")
    with open(args.test_file, 'r') as f:
        # It's JSONL
        data = [json.loads(line) for line in f]
    
    correct = 0
    total = 0
    hallucinations = 0
    
    results = []
    
    print(f"Evaluating on {len(data)} examples...")
    for item in tqdm(data):
        prompt = item['prompt']
        ground_truth = item['chosen']
        
        # Extract state description from prompt for generation
        # Prompt format: "... Current State: {desc} \n Next Subgoal:"
        # Actually Phi2Planner constructs the prompt internally from state_description.
        # So we need to parse the state description back out from the prompt, 
        # OR just use the raw 'prompt' directly if we modify Planner to accept raw prompt?
        # A cleaner way is to load 'expert_traces_test.json' which has 'state_text'.
        # But 'dpo_test.jsonl' has 'prompt'.
        # Let's simple parse properties from the prompt string since it's structured.
        
        try:
             # Hacky extraction matching PROMPT_TEMPLATE
             state_desc = prompt.split("Current State:")[-1].split("Next Subgoal:")[0].strip()
        except:
             state_desc = prompt # Fallback
             
        # Generate
        prediction = planner.generate_subgoal(state_desc)
        
        # Check correctness
        is_correct = (prediction.strip().lower() == ground_truth.strip().lower())
        
        # Check structure (Hallucination check)
        # Should be short string, should not contain "Objective:" etc.
        is_hallucination = len(prediction) > 100 or "Objective:" in prediction
        
        if is_correct: correct += 1
        if is_hallucination: hallucinations += 1
        
        total += 1
        
        results.append({
            "state": state_desc,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": is_correct
        })
        
    accuracy = correct / total
    hall_rate = hallucinations / total
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Hallucination Rate: {hall_rate:.2%}")
    
    # Save log
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapters")
    parser.add_argument("--test_file", type=str, default="data/dpo_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="results_eval")
    parser.add_argument("--load_in_4bit", action="store_true")
    
    args = parser.parse_args()
    evaluate_alignment(args)
