import json
import jsonlines
import os

def verify_file(filepath):
    print(f"\nVerifying {filepath}...")
    if not os.path.exists(filepath):
        print("❌ File not found.")
        return False
        
    errors = 0
    count = 0
    unique_prompts = set()
    
    try:
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                count += 1
                
                # Check fields
                if "prompt" not in obj or "chosen" not in obj or "rejected" not in obj:
                    print(f"❌ Line {count}: Missing keys.")
                    errors += 1
                    continue
                
                prompt = obj['prompt']
                chosen = obj['chosen']
                rejected = obj['rejected']
                
                # Check empty
                if not prompt or not chosen or not rejected:
                    print(f"❌ Line {count}: Empty values.")
                    errors += 1
                    continue
                    
                # Check identity
                if chosen == rejected:
                    print(f"❌ Line {count}: Chosen == Rejected ('{chosen}')")
                    errors += 1
                    continue
                
                # Check ambiguity (Soft check)
                # E.g., "Pick the key" vs "Pick up key" - might be acceptable but worth noting.
                # For our canonical parser, they should be distinct enough.
                
                unique_prompts.add(prompt)
                
    except Exception as e:
        print(f"❌ Failed to parse JSONL: {e}")
        return False
        
    print(f"✔ Scanned {count} examples.")
    print(f"✔ Unique prompts: {len(unique_prompts)}")
    
    if errors == 0:
        print("✅ Integrity Check Passed.")
        return True
    else:
        print(f"❌ Integrity Check Failed with {errors} errors.")
        return False

if __name__ == "__main__":
    files = [
        "data/dpo_train.jsonl",
        "data/dpo_val.jsonl",
        "data/dpo_test.jsonl"
    ]
    
    all_pass = True
    for f in files:
        if not verify_file(f):
            all_pass = False
            
    if all_pass:
        print("\n✅ GLOBAL CHECK PASSED")
    else:
         print("\n❌ GLOBAL CHECK FAILED")
