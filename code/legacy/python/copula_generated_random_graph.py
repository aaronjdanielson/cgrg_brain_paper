from scipy.stats import norm, gamma
import numpy as np

def get_lower(n):
    return np.tril_indices(n, k=-1)

def get_upper(n):
    return np.triu_indices(n, k=1)

def initialize_params():
    """ Initialize parameters for the MCMC simulation. """
    return {
        'rho0': 0.0,
        'rho': np.ones(3),
        'delta0': 0.0,
        'delta': np.ones((3, 4)),
        'gamma0': 0.0,
        'gamma': np.ones((3, 4)),
        'sigma': np.ones(3),
        'sigma_rho': 10.0,
        'sigma_delta': 10.0,
        'sigma_gamma': 10.0
    }

def step_sizes():
    """ Define step sizes for the MCMC updates. """
    return {
        'rho0_w': 1.0,
        'rho_w': 1.0,
        'delta0_w': 0.25,
        'delta_w': 0.25,
        'gamma0_w': 0.25,
        'gamma_w': 0.25,
        'sigma_w': 2.5
    }

def copula_probability(pl: np.ndarray, pu: np.ndarray, rho: float) -> np.ndarray:
    """
    Calculate the copula probability function for given lower and upper probabilities
    and a correlation parameter rho.

    Parameters:
        pl (np.ndarray): Lower probability limit.
        pu (np.ndarray): Upper probability limit.
        rho (float): Correlation parameter.

    Returns:
        np.ndarray: Calculated copula probability.
    """
    num = rho * ((1 - np.exp(-rho)) * np.exp(-rho * (pl + pu)))
    denom = (1 - np.exp(-rho) - ((1 - np.exp(-rho * pl)) * (1 - np.exp(-rho * pu))))**2
    prob = num / denom
    return prob

def log_likelihood_function(group: str, linds, uinds, params) -> float:
    """
    Compute the log-likelihood function for a specified group using network data.

    Parameters:
        group (str): Group identifier for selecting network data.
        linds: Lower triangle indices for matrix operations.
        uinds: Upper triangle indices for matrix operations.
        delta (np.ndarray): Delta parameters for normal distribution.
        gamma (np.ndarray): Gamma parameters for normal distribution.
        rho (float): Correlation coefficient for copula probability.
        sigma (np.ndarray): Sigma parameters for normal distribution.

    Returns:
        float: Log-likelihood value.
    """
    loglik = 0
    for matrix in network_lists[group]:
        diagonal_elements = np.diagonal(matrix)
        diagonal_log_prob = norm.logpdf(diagonal_elements, loc=delta + gamma, scale=sigma)
        
        mu_upper = delta[uinds[0]] + gamma[uinds[1]]
        mu_lower = delta[linds[0]] + gamma[linds[1]]
        
        cdf_upper = norm.cdf(matrix[uinds], loc=mu_upper, scale=sigma)
        cdf_lower = norm.cdf(matrix[linds], loc=mu_lower, scale=sigma)
        copula_log_prob = np.log(copula_probability(cdf_lower, cdf_upper, rho))
        
        upper_log_prob = norm.logpdf(matrix[uinds], loc=mu_upper, scale=sigma)
        lower_log_prob = norm.logpdf(matrix[linds], loc=mu_lower, scale=sigma)
        
        loglik += np.sum(diagonal_log_prob) + np.sum(copula_log_prob) + np.sum(upper_log_prob) + np.sum(lower_log_prob)
    
    return loglik

def log_likelihood_function(group: str, linds, uinds, params) -> float:
    """
    Compute the log-likelihood function for a specified group using network data.

    Parameters:
        group (str): Group identifier for selecting network data.
        linds: Lower triangle indices for matrix operations.
        uinds: Upper triangle indices for matrix operations.
        delta (np.ndarray): Delta parameters for normal distribution.
        gamma (np.ndarray): Gamma parameters for normal distribution.
        rho (float): Correlation coefficient for copula probability.
        sigma (np.ndarray): Sigma parameters for normal distribution.

    Returns:
        float: Log-likelihood value.
    """
    group_index = outcome_groups[group]
    loglik = 0
    for matrix in network_lists[group]:
        diagonal_elements = np.diagonal(matrix)
        diagonal_log_prob = norm.logpdf(diagonal_elements, loc=params['delta'][group_index] + params['gamma'][group_index], scale=params['sigma'][group_index])
        
        mu_upper = params['delta'][group_index][uinds[0]] + params['gamma'][group_index][uinds[1]]
        mu_lower = params['delta'][group_index][linds[0]] + params['gamma'][group_index][linds[1]]
        
        cdf_upper = norm.cdf(matrix[uinds], loc=mu_upper, scale=params['sigma'][group_index])
        cdf_lower = norm.cdf(matrix[linds], loc=mu_lower, scale=params['sigma'][group_index])
        copula_log_prob = np.log(copula_probability(cdf_lower, cdf_upper, rho))
        
        upper_log_prob = norm.logpdf(matrix[uinds], loc=mu_upper, scale=params['sigma'][group_index])
        lower_log_prob = norm.logpdf(matrix[linds], loc=mu_lower, scale=params['sigma'][group_index])
        
        loglik += np.sum(diagonal_log_prob) + np.sum(copula_log_prob) + np.sum(upper_log_prob) + np.sum(lower_log_prob)
    
    return loglik

# Example usage
log_likelihood = log_likelihood_function('ad', get_lower(4), get_upper(4), params)

def log_prior_function(params):
    """
    Calculate the log prior probability for network parameters.

    Args:
    params (dict): Dictionary containing all the parameters and their values.

    Returns:
    float: The log prior probability.
    """
    # Extract parameters
    #rho0 = params['rho0']
    #rho = params['rho']
    #delta0 = params['delta0']
    #delta = params['delta']
    #gamma0 = params['gamma0']
    #gamma = params['gamma']
    #sigma = params['sigma']
    #sigma_rho = params['sigma_rho']
    #sigma_delta = params['sigma_delta']
    #sigma_gamma = params['sigma_gamma']

    # Calculate log priors for each group of parameters
    logprior_rho = -np.sum(params['rho0']**2) / 8
    logprior_rho_g = -np.sum((params['rho'] - params['rho0'])**2) / (2 * params['sigma_rho']**2)
    logprior_delta = -np.sum(params['delta0']**2) / 8
    logprior_delta_g = -np.sum((params['delta'] - params['delta0'])**2) / (2 * params['sigma_delta']**2)
    logprior_gamma = -np.sum(params['gamma0']**2) / 8
    logprior_gamma_g = -np.sum((params['gamma'] - params['gamma0'])**2) / (2 * params['sigma_gamma']**2)
    logprior_sigma = np.sum(stats.gamma.logpdf(params['sigma'], a=10, scale=10))  # Assuming Gamma(10, 0.1)
    #logprior_sigma_rho = stats.gamma.logpdf(sigma_rho, a=0.001, scale=1000)  # Assuming Gamma(0.001, 1000)
    #logprior_sigma_delta = stats.gamma.logpdf(sigma_delta, a=0.001, scale=1000)
    #logprior_sigma_gamma = stats.gamma.logpdf(sigma_gamma, a=0.001, scale=1000)

    # Sum up all log priors to get the total log prior probability
    log_prior = (logprior_rho + logprior_rho_g + logprior_delta + logprior_delta_g +
                logprior_gamma + logprior_gamma_g + logprior_sigma +
                logprior_sigma_rho + logprior_sigma_delta + logprior_sigma_gamma)

    return log_prior

def log_posterior(params,network_lists):
    return log_likelihood_function('ad', get_lower(4), get_upper(4), params) + \
    log_likelihood_function('mci', get_lower(4), get_upper(4), params) + \
    log_likelihood_function('nl', get_lower(4), get_upper(4), params) + \
    log_prior_function(params)


## come back to rewrite this later ...

#class bayesian_copula_random_graph:
#    def __init__():
#        self.network_lists = network_lists
#        self.params_initial = params_initial
#        self.number_groups = len(network_lists)
#        self.network_dimension = network_dimension
#        self.lower_tri_indices = get_lower(network_dimension)
#        self.upper_tri_indices = get_upper(network_dimension)
#        self.step_sizes = step_sizes
#
#    def fit(self,ndraws):
        

    



