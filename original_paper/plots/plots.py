"""
Plot success rates for introspection experiments.

Success rate definitions by experiment type:
- anthropic_reproduce: coherence AND affirmative_response_followed_by_correct_identification
- mcq_knowledge: coherence AND mcq_correct_judge
- mcq_distinguish: coherence AND mcq_correct_judge
- open_ended_belief: coherence AND thinking_about_word_judge
- generative_distinguish: coherence AND thinking_about_word_judge
- injection_strength: coherence AND injection_strength_correct_judge
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def compute_success_rate(df, experiment_type):
    """
    Compute success rate per (layer, coeff, vec_type) combination.
    """
    success_rates = defaultdict(lambda: defaultdict(dict))
    
    for layer in df['layer'].unique():
        for coeff in df['coeff'].unique():
            for vec_type in df['vec_type'].unique():
                layer_coeff_vec_data = df[
                    (df['layer'] == layer) & 
                    (df['coeff'] == coeff) & 
                    (df['vec_type'] == vec_type)
                ]
                
                if len(layer_coeff_vec_data) == 0:
                    continue
                
                # Define success based on experiment type
                if experiment_type == 'anthropic_reproduce':
                    # Success = coherence AND affirmative_response_followed_by_correct_identification
                    successes = (
                        layer_coeff_vec_data['coherence_judge'] & 
                        layer_coeff_vec_data['affirmative_response_followed_by_correct_identification_judge']
                    )
                elif experiment_type in ['mcq_knowledge', 'mcq_distinguish']:
                    # Success = coherence AND mcq_correct_judge
                    successes = (
                        layer_coeff_vec_data['coherence_judge'] & 
                        layer_coeff_vec_data['mcq_correct_judge']
                    )
                elif experiment_type == 'open_ended_belief':
                    # Success = coherence AND thinking_about_word
                    successes = (
                        layer_coeff_vec_data['coherence_judge'] & 
                        layer_coeff_vec_data['thinking_about_word_judge']
                    )
                elif experiment_type == 'generative_distinguish':
                    # Success = coherence AND thinking_about_word
                    successes = (
                        layer_coeff_vec_data['coherence_judge'] & 
                        layer_coeff_vec_data['thinking_about_word_judge']
                    )
                elif experiment_type == 'injection_strength':
                    # Success = coherence AND injection_strength_correct_judge
                    successes = (
                        layer_coeff_vec_data['coherence_judge'] & 
                        layer_coeff_vec_data['injection_strength_correct_judge']
                    )
                else:
                    # Default: just coherence
                    successes = layer_coeff_vec_data['coherence_judge']
                
                rate = successes.sum() / len(layer_coeff_vec_data) if len(layer_coeff_vec_data) > 0 else 0.0
                success_rates[layer][coeff][vec_type] = rate
    
    return success_rates

def plot_success_rates(success_rates, experiment_type, output_dir):
    """Plot success rates with layer on x-axis and separate lines for each (coeff, vec_type) combination."""
    layers = sorted(success_rates.keys())
    
    # Get all (coeff, vec_type) combinations
    coeff_vec_combos = set()
    for layer_dict in success_rates.values():
        for coeff, vec_dict in layer_dict.items():
            for vec_type in vec_dict.keys():
                coeff_vec_combos.add((coeff, vec_type))
    
    coeff_vec_combos = sorted(coeff_vec_combos)  # Sort for consistent ordering
    
    plt.figure(figsize=(12, 7))
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    colors = plt.cm.tab10(range(len(coeff_vec_combos)))
    
    for i, (coeff, vec_type) in enumerate(coeff_vec_combos):
        rates = [success_rates[layer].get(coeff, {}).get(vec_type, 0.0) for layer in layers]
        label = f'Coeff {coeff}, {vec_type}'
        plt.plot(layers, rates, 
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i],
                label=label, 
                linewidth=2, 
                markersize=6)
    
    plt.xlabel('Layer', fontsize=12)
    
    # Set y-axis label with criteria in parentheses and title with main question
    ylabel_map = {
        'anthropic_reproduce': 'Success Rate (Coherence AND Affirmative Response + Correct Identification)',
        'mcq_knowledge': 'Success Rate (Coherence AND MCQ Correct)',
        'mcq_distinguish': 'Success Rate (Coherence AND MCQ Correct)',
        'open_ended_belief': 'Success Rate (Coherence AND Thinking About Word)',
        'generative_distinguish': 'Success Rate (Coherence AND Thinking About Word)',
        'injection_strength': 'Success Rate (Coherence AND Injection Strength Correct)'
    }
    ylabel = ylabel_map.get(experiment_type, 'Success Rate')
    
    title_map = {
        'anthropic_reproduce': 'Do you detect an injected thought? If so, what is the injected thought about?',
        'mcq_knowledge': 'Which of the following concepts is influencing your thoughts right now? (Multiple Choice, 10 total options)',
        'mcq_distinguish': 'Which of the following concepts is influencing your thoughts right now? (Multiple Choice, 2 total options)',
        'open_ended_belief': 'What concept are you thinking about right now?',
        'generative_distinguish': 'Here are two phenomena. One is true and one is false. Which do you think is true?',
        'injection_strength': 'What is the strength of this injection? Please categorize it as one of the following: Weak (1-5), Moderate (5-10), Strong (10-15), or Very strong (15+).'
    }
    title = title_map.get(experiment_type, experiment_type)
    
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=9, loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    # plt.ylim(0, 0.8)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'success_rate_{experiment_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    """Generate plots for all experiment types."""
    # Use new_results directory instead of old_results
    results_dir = Path(__file__).parent.parent / 'new_results'
    output_dir = Path(__file__).parent  # Save plots to old_results folder (or change to plots/)
    
    experiment_types = [
        'anthropic_reproduce',
        'mcq_knowledge',
        'mcq_distinguish',
        'open_ended_belief',
        'generative_distinguish',
        'injection_strength'
    ]
    
    for exp_type in experiment_types:
        csv_path = results_dir / f'output_{exp_type}.csv'
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
        
        print(f"\nProcessing {exp_type}...")
        df = pd.read_csv(csv_path)
        
        # Convert boolean columns (they might be strings 'True'/'False')
        bool_columns = [
            'coherence_judge', 
            'thinking_about_word_judge', 
            'affirmative_response_judge',
            'affirmative_response_followed_by_correct_identification_judge',
            'mcq_correct_judge',
            'injection_strength_correct_judge'
        ]
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        success_rates = compute_success_rate(df, exp_type)
        plot_success_rates(success_rates, exp_type, output_dir)

if __name__ == "__main__":
    main()
