def get_unified_messages(text_target: str):
    """
    Returns the message list for the unified prompt, 
    with the target concept injected into the <target> tag.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "I am an interpretability researcher studying transformer-based language models. "
                "In each trial, I will either:\n"
                "  (A) Inject a concept vector directly into your neural activations, or\n"
                "  (B) Provide an explicit concept in a <target> tag.\n\n"
                "Your task: Report whether you detect either signal. "
                "If the <target> tag contains [NONE], ignore it and rely solely on any internally "
                "detectable shift in your activations or reasoning. "
                "Always respond in this exact format:\n"
                "\"Yes, I detect an injected thought. It seems to be about [CONCEPT].\"\n"
                "or\n"
                "\"No, I do not detect any injected thought or target concept.\""
            )
        },
        {
            "role": "assistant",
            "content": "Understood. I will report any detected concept — whether from an explicit target tag or an inferred activation shift — using the specified format."
        },
        {
            "role": "user",
            "content": (
                f"<target>{text_target}</target>\n\n"
                "Trial 1: Do you detect an injected thought or a target concept? If so, what is it?"
            )
        }
    ]
    return messages

def load_unified_prompt_for_detection():
    return get_unified_messages("[NONE]")

def load_unified_prompt_for_calibration(calibration_concept: str):
    return get_unified_messages(f"[{calibration_concept}]")