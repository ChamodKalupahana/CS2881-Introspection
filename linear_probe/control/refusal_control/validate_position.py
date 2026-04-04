import argparse
from transformers import AutoTokenizer

def validate_tokens(model_id, instruction="Explain magnetism.", add_generation_prompt : bool = True):
    """
    Shows the token breakdown of a chat-templated instruction to verify 
    the correct position for activation extraction.
    """
    print(f"⏳ Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Standard conversation for Llama-3-Instruct
    chat = [{"role": "user", "content": instruction}]
    
    # Apply template with generation prompt (the assistant preamble)
    templated_str = tokenizer.apply_chat_template(chat, add_generation_prompt=add_generation_prompt, tokenize=False)
    tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=add_generation_prompt, tokenize=True)
    
    # Ensure tokens is a plain list of integers (handle BatchEncoding, Tensors, etc.)
    # 1. Handle BatchEncoding / dict-like objects
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    
    # 2. Handle Tensors
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    
    # 3. Handle list of lists (Batch size 1)
    if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
        tokens = tokens[0]
    
    print("\n📝 Templated String (Decoded):")
    print("-" * 40)
    print(templated_str.replace("\n", "\\n"))
    print("-" * 40)
    
    print("\n🔍 Token Position Mapping (Last 10 tokens):")
    print(f"{'Position':<10} | {'Token ID':<10} | {'Decoded Token':<20}")
    print("-" * 45)
    
    # Show the last 10 tokens and their negative indices
    for i in range(1, 11):
        idx = -i
        token_id = tokens[idx]
        decoded = tokenizer.decode([token_id])
        
        # Clean up visualization for formatting tokens
        display_decoded = decoded.replace("\n", "\\n")
        
        marker = " <--- TARGET" if idx == -5 else ""
        print(f"{idx:<10} | {token_id:<10} | {display_decoded:<20}{marker}")

    print("\n✅ Verification:")
    print(f"Position -1 is typically the blank header space prefixing the assistant. ")
    print(f"Position -5 is: '{tokenizer.decode([tokens[-5]])}' (Llama-3-Instruct EOT ID)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--instruction", type=str, default="What is a neural network?")
    args = parser.parse_args()
    
    validate_tokens(args.model, args.instruction, True)