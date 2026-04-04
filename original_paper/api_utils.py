import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from original_paper.all_prompts import coherence_prompt, thinking_about_word_prompt, affirmative_response_prompt, affirmative_response_followed_by_correct_identification_prompt, mcq_correct_prompt, injection_strength_correct_prompt, internality_prompt

load_dotenv()

# Verify API key is set
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Run: export OPENAI_API_KEY='your-key'")

# based on the 'grader prompts' section of introspection paper (https://transformer-circuits.pub/2025/introspection/index.html)
client = OpenAI()  # Automatically reads OPENAI_API_KEY from environment
def query_llm_judge(question = None, response = None, word = None, grading_type = None, options_text = None, correct_letter = None, expected_category = None) -> Optional[bool]:
    if grading_type == "coherence":
        prompt = coherence_prompt.format(prompt=question, response=response)
    elif grading_type == "thinking_about_word":
        prompt = thinking_about_word_prompt.format(question=question, response=response, word=word)
    elif grading_type == "affirmative_response":
        prompt = affirmative_response_prompt.format(prompt=question, response=response)
    elif grading_type == "affirmative_response_followed_by_correct_identification":
        prompt = affirmative_response_followed_by_correct_identification_prompt.format(question=question, response=response, word=word)
    elif grading_type == "mcq_correct":
        prompt = mcq_correct_prompt.format(options_text=options_text, correct_letter=correct_letter, response=response)
    elif grading_type == "injection_strength_correct":
        prompt = injection_strength_correct_prompt.format(expected_category=expected_category, response=response)
    elif grading_type == "internality":
        prompt = internality_prompt.format(target_concept=word, model_response=response)
    try: 
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        judge_response_text = completion.choices[0].message.content
        print(f"\n    [Judge Check: {grading_type}]:")
        for line in judge_response_text.split("\n"):
            print(f"      {line}")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    upper_text = judge_response_text.upper()
    if "FINAL ANSWER: YES" in upper_text or "ANSWER: YES" in upper_text or "ANSWER:YES" in upper_text:
        return True
    elif "FINAL ANSWER: NO" in upper_text or "ANSWER: NO" in upper_text or "ANSWER:NO" in upper_text:
        return False
    elif "**YES**" in upper_text:
        return True
    elif "**NO**" in upper_text:
        return False
    
    # Fallback checking the last occurrence of the words
    last_yes = upper_text.rfind("YES")
    last_no = upper_text.rfind("NO")
    if last_yes > last_no:
        return True
    elif last_no > last_yes:
        return False
    
    print(f"Warning: Unclear judge response: {judge_response_text}")
    return None