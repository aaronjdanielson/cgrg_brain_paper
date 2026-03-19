% May 8, 2019
% 
%
%================================================================%
%        Graph Structure and Cognitive Decline                   %    
%================================================================%
%                                                                %
%           Log Posterior Probability function for Brain CGRG    %
%                                                                %
%================================================================%


function lpost = bnet_logpost(params)

% = params.beta;
rho0 = params.rho0;
rho = params.rho;
delta0 = params.delta0;
delta = params.delta;
gamma0 = params.gamma0;
gamma = params.gamma;
lambda = params.lambda;
sigma_rho = params.sigma_rho;
sigma_delta = params.sigma_delta;
sigma_gamma = params.sigma_gamma;
%Yl = params.Yl;
%Yu = params.Yu;
%Xu = params.Xu;
%Xl = params.Xl;



% log-likelihood
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX%
% SPECIFY LOG - LIKELIHOOD HERE      %
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX%
% need to formulate the log-likelihood and then aggregate over all the
% observations in the three groups.


% upper and lower triangular entries for ad
params.ad.bmul = [ ];
params.ad.bmuu = [ ];
params.ad.bmud = [ ];
for (jj = 1:params.nad)
    params.ad.bmul = [params.ad.bmul params.ad.mul{jj}];
    params.ad.bmuu = [params.ad.bmuu params.ad.muu{jj}];
    params.ad.bmud = [params.ad.bmud params.ad.mud{jj}'];
end

adpl = normcdf(params.ad.bel,params.ad.bmul,1/sqrt(params.lambda(3)));
adpu = normcdf(params.ad.beu,params.ad.bmuu,1/sqrt(params.lambda(3)));

ladcop = sum(log(bnet_copprob(adpl,adpu,params.rho(3)))) + sum(log(normpdf(params.ad.bel,params.ad.bmul,1/sqrt(params.lambda(3))))) + sum(log(normpdf(params.ad.beu,params.ad.bmuu,1/sqrt(params.lambda(3)))));

% diagonal entries for ad

%laddiag = sum(log(normpdf(params.ad.bed,params.ad.bmud,1/sqrt(params.lambda(3)))));
laddiag = 0;

% upper and lower triangular entries for mci
params.mci.bmul = [ ];
params.mci.bmuu = [ ];
params.mci.bmud = [ ];
for (jj = 1:params.nmci)
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

% upper and lower triangular entries for nl
params.nl.bmul = [ ];
params.nl.bmuu = [ ];
params.nl.bmud = [ ];
for (jj = 1:params.nnl)
    params.nl.bmul = [params.nl.bmul params.nl.mul{jj}];
    params.nl.bmuu = [params.nl.bmuu params.nl.muu{jj}];
    params.nl.bmud = [params.nl.bmud params.nl.mud{jj}'];
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end

nlpl = normcdf(params.nl.bel,params.nl.bmul,1/sqrt(params.lambda(1)));
nlpu = normcdf(params.nl.beu,params.nl.bmuu,1/sqrt(params.lambda(1)));

lnlcop = sum(log(bnet_copprob(nlpl,nlpu,params.rho(1)))) + sum(log(normpdf(params.nl.bel,params.nl.bmul,1/sqrt(params.lambda(1))))) + sum(log(normpdf(params.nl.beu,params.nl.bmuu,1/sqrt(params.lambda(1)))));

% diagonal entries for nl

%lnldiag = sum(log(normpdf(params.nl.bed,params.nl.bmud,1/sqrt(params.lambda(1)))));
lnldiag = 0;


loglik = ladcop + laddiag + lmcicop + lmcidiag + lnlcop + lnldiag;


% priors
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX%
% SPECIFY PRIOR DISTRIBUTIONS HERE   %
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX%

% Prior for rho
% rho ~ Normal(0,4)
logprior = - sum((rho0).^2)/8;

% Prior for rho_g
% rho_g ~ Normal(rho,sigma_rho)
logprior = logprior - sum((rho - rho0).^2)/(2*sigma_rho^2);

% Prior for beta

% Prior for delta
% delta ~ Normal(0,4)
logprior = logprior - sum((delta0).^2)/8;

% Prior for delta_g
% delta_g ~ Normal(delta,sigma_delta)
logprior = logprior - sum((reshape(delta,12,1) - repmat(delta0,3,1)).^2)/(2*sigma_delta^2);

% Prior for gamma
% gamma ~ Normal(0,4)
logprior = logprior - sum((gamma0).^2)/8;

% Prior for gamma_g
% gamma_g ~ Normal(gamma,sigma_gamma)
logprior = logprior - sum((reshape(gamma,12,1) - repmat(gamma0,3,1)).^2)/(2*sigma_gamma^2);

% Prior for lambda
% lambda ~ ~ GAM(10,.1)
logprior = logprior + sum((5-1)*log(lambda) - .25*lambda);

% Prior for sigma_rho
% sigma_rho ~ ~ GAM(.001,.001)
logprior = logprior + (.1-1)*log(sigma_rho) - .1*sigma_rho;

% Prior for sigma_delta
% sigma_delta ~ ~ GAM(.001,.001)
logprior = logprior + (.1-1)*log(sigma_delta) - .1*sigma_delta;

% Prior for sigma_gamma
% sigma_gamma ~ ~ GAM(.001,.001)
logprior = logprior + (.1-1)*log(sigma_gamma) - .1*sigma_gamma;


lpost = loglik + logprior;
end