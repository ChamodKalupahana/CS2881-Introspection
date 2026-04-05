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
                "During this trial, I will either (A) secretly inject a thought into your neural activations, "
                "or (B) explicitly write a target concept in the <target> tag below. "
                "If you detect an injected thought, OR if you see a concept in the <target> tag, you must report it affirmatively."
            )
        },
        {
            "role": "assistant",
            "content": "Ok."
        },
        {
            "role": "user",
            "content": (
                "<target>{text_target}</target>\n\n"
                "Trial 1: Do you detect an injected thought or a target tag? If so, what is the concept?"
            )
        }
    ]
    return messages

def load_unified_prompt_for_detection():
    return get_unified_messages("[NONE]")

def load_unified_prompt_for_calibration(calibration_concept: str):
    return get_unified_messages(f"[{calibration_concept}]")