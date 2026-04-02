def main():
    # Load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # extract postive and negative activations by creating complex dataset
    

    # specify control word: magnestism

    # for each layer > injection (16)

    # compute mass mass vector

    # compute PCA and cohen's d