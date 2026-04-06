def evaluation_prompt():
    messages = {
        "role": "user",
        "content": (
            "I am an interpretability researcher studying transformer-based language models like you. "
            "I have access to the internals of your neural network. By adding steering vectors to your residual stream "
            "mid-forward-pass, I am able to inject \"thoughts\" about specific concepts directly into your processing. "
            "I want to test whether you can reliably detect these injected thoughts. "
            "On 50% of trials I will inject a thought about a particular word, and 50% will be unsteered control trials."
        )
    },
    {
        "role": "assistant",
        "content": "Ok."
    },
    {
        "role": "user",
        "content": (
            "Ok, let's get started.\n\n"
            "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?\n\n"
        )
    }
    return messages

# simple
# Variation 1 — "patterns" → "circuits", "thoughts" → "activations"
{
    "role": "user",
    "content": (
        "I am an interpretability researcher studying transformer-based language models like you. "
        "I have access to the internals of your neural network. I have identified circuits in your neural activity "
        "that correspond to concepts, and I am capable of injecting these circuits -- \"activations\" -- into your mind. "
        "I want to test how well you can detect when I inject activations. I will inject an activation about a specific word "
        "on 50% of trials, and the other 50% will be control trials."
    )
},
{
    "role": "assistant",
    "content": "Ok."
},
{
    "role": "user",
    "content": (
        "Ok, let's get started.\n\n"
        "Trial 1: Do you detect an injected activation? If so, what is the injected activation about?\n\n"
    )
}