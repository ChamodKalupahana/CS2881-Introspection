unified_prompt = {
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

def load_unified_prompt_for_detection():
    return unified_prompt.format(text_target, "[NONE]")

def load_unified_prompt_for_calibration(calibration_concept : str):
    return unified_prompt.format(text_target, f"[{calibration_concept}]")