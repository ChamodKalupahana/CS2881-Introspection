import numpy as np

def compute_cohens_d(x1, x2):
    """
    Computes Cohen's d for two independent samples.
    Args:
        x1: Array of values from distribution 1.
        x2: Array of values from distribution 2.
    Returns:
        d: Cohen's d effect size.
    """
    m1, m2 = np.mean(x1), np.mean(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1) # Sample standard deviations
    
    # Pooled standard deviation
    n1, n2 = len(x1), len(x2)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # If pooled std is very small, return 0 to avoid division by zero
    if pooled_std < 1e-9:
        return 0.0
        
    return abs(m1 - m2) / pooled_std