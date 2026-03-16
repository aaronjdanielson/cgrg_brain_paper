import numpy as np

def bnet_copprob(pl, pu, rho):
    """
    Calculate the copula probability function for given lower and upper probabilities
    and a correlation parameter rho.
    
    Parameters:
        pl (float): Lower probability limit.
        pu (float): Upper probability limit.
        rho (float): Correlation parameter.
        
    Returns:
        float: Calculated copula probability.
    """
    num = rho * ((1 - np.exp(-rho)) * np.exp(-rho * (pl + pu)))
    denom = (1 - np.exp(-rho) - ((1 - np.exp(-rho * pl)) * (1 - np.exp(-rho * pu))))**2
    prob = num / denom
    return prob

# Example usage
pl = 0.2
pu = 0.5
rho = 0.7
prob = bnet_copprob(pl, pu, rho)
print("Copula Probability:", prob)
