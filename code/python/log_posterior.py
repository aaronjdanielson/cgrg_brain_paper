import numpy as np
from scipy.stats import norm, gamma

def bnet_logpost(params):
    # Extract parameters
    rho0 = params['rho0']
    rho = params['rho']
    delta0 = params['delta0']
    delta = params['delta']
    gamma0 = params['gamma0']
    gamma = params['gamma']
    lambda_ = params['lambda']
    sigma_rho = params['sigma_rho']
    sigma_delta = params['sigma_delta']
    sigma_gamma = params['sigma_gamma']


    # upper and lower triangular entries for nl
    params.nl.bmul = [ ];
    params.nl.bmuu = [ ];
    params.nl.bmud = [ ];
    for (jj = 1:params.nnl):
        params.nl.bmul = [params.nl.bmul params.nl.mul{jj}];
        params.nl.bmuu = [params.nl.bmuu params.nl.muu{jj}];
        params.nl.bmud = [params.nl.bmud params.nl.mud{jj}'];
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
    end

    nlpl = normcdf(params.nl.bel,params.nl.bmul,1/sqrt(params.lambda(1)));
    nlpu = normcdf(params.nl.beu,params.nl.bmuu,1/sqrt(params.lambda(1)));

    lnlcop = sum(log(bnet_copprob(nlpl,nlpu,params.rho(1)))) + sum(log(normpdf(params.nl.bel,params.nl.bmul,1/sqrt(params.lambda(1))))) + sum(log(normpdf(params.nl.beu,params.nl.bmuu,1/sqrt(params.lambda(1)))));

% diagonal entries for nl

    slnldiag = sum(log(normpdf(params.nl.bed,params.nl.bmud,1/sqrt(params.lambda(1)))));
lnldiag = 0

    ## upper and lower triangular entries for mci
    params.mci.bmul = [ ];
    params.mci.bmuu = [ ];
    params.mci.bmud = [ ];
    for (jj = 1:params.nmci):
        params.mci.bmul = [params.mci.bmul params.mci.mul{jj}];
        params.mci.bmuu = [params.mci.bmuu params.mci.muu{jj}];
        params.mci.bmud = [params.mci.bmud params.mci.mud{jj}'];
        %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end

    mcipl = normcdf(params.mci.bel,params.mci.bmul,1/sqrt(params.lambda(2)));
    mcipu = normcdf(params.mci.beu,params.mci.bmuu,1/sqrt(params.lambda(2)));

    lmcicop = sum(log(bnet_copprob(mcipl,mcipu,params.rho(2)))) + sum(log(normpdf(params.mci.bel,params.mci.bmul,1/sqrt(params.lambda(2))))) + sum(log(normpdf(params.mci.beu,params.mci.bmuu,1/sqrt(params.lambda(2)))));

    % diagonal entries for mci

    %lmcidiag = sum(log(normpdf(params.mci.bed,params.mci.bmud,1/sqrt(params.lambda(2)))));
    lmcidiag = 0;


    # Log-likelihood components
    # Here you would calculate your model's log-likelihood based on the model design
    # This will involve calculating probabilities from your data using the model parameters
    # For simplicity, let's assume it is pre-calculated or define a placeholder
    loglik = ladcop + laddiag + lmcicop + lmcidiag + lnlcop + lnldiag # Placeholder, replace with actual log-likelihood calculation

    # Priors
    logprior_rho = -np.sum(rho0**2) / 8
    logprior_rho_g = -np.sum((rho - rho0)**2) / (2 * sigma_rho**2)
    logprior_delta = -np.sum(delta0**2) / 8
    logprior_delta_g = -np.sum((delta - delta0)**2) / (2 * sigma_delta**2)
    logprior_gamma = -np.sum(gamma0**2) / 8
    logprior_gamma_g = -np.sum((gamma - gamma0)**2) / (2 * sigma_gamma**2)
    logprior_lambda = gamma.logpdf(lambda_, a=10, scale=10)  # Assuming Gamma(10, 0.1)
    logprior_sigma_rho = gamma.logpdf(sigma_rho, a=0.001, scale=1000)  # Assuming Gamma(0.001, 1000)
    logprior_sigma_delta = gamma.logpdf(sigma_delta, a=0.001, scale=1000)
    logprior_sigma_gamma = gamma.logpdf(sigma_gamma, a=0.001, scale=1000)

    # Total log prior
    logprior = (logprior_rho + logprior_rho_g + logprior_delta + logprior_delta_g +
                logprior_gamma + logprior_gamma_g + logprior_lambda +
                logprior_sigma_rho + logprior_sigma_delta + logprior_sigma_gamma)

    # Log posterior
    lpost = loglik + logprior
    return lpost

# Example of using this function
params = {
    'rho0': np.array([0.1, 0.2]),
    'rho': np.array([0.1, 0.25]),
    'delta0': np.array([0.1]),
    'delta': np.array([0.15]),
    'gamma0': np.array([0.1]),
    'gamma': np.array([0.12]),
    'lambda': np.array([10]),
    'sigma_rho': 0.5,
    'sigma_delta': 0.5,
    'sigma_gamma': 0.5
}

log_post = bnet_logpost(params)
print("Log Posterior:", log_post)
