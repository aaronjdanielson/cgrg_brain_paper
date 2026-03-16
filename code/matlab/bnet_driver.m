% May 8, 2019
% 
%
%================================================================%
%        Graph Structure and Cognitive Decline                   %    
%================================================================%
%                                                                %
%           Driver function for Brain CGRG                       %
%                                                                %
%================================================================%

function [pvals, params] = bnet_driver(ad,mci,nl,niter)

linds = get_lower(4);
uinds = get_upper(4);


% shape the networks

% create the parameters

params.nad = 12;
params.nmci = 63;
params.nnl = 37;
params.rho0 = 0.00001;
params.rho = zeros(3,1) + .000001;

params.delta0 = zeros(4,1) + .0000;
params.delta = zeros(4,3) + .0000;
%params.delta = rand(4,3);

params.gamma0 = zeros(4,1) + .0000;
params.gamma = zeros(4,3) + .0000;
%params.gamma = rand(4,3);

params.lambda = zeros(3,1) + 5;
params.sigma_rho = 1;
params.sigma_delta = 1;
params.sigma_gamma = 1;

% get step sizes
params.rho0_w = 1;
params.rho_w = zeros(3,1) + 1;

params.delta0_w = zeros(4,1) + .25;
params.delta_w = zeros(4,3) + .25;
%params.delta = rand(4,3);

params.gamma0_w = zeros(4,1) + .25;
params.gamma_w = zeros(4,3) + .25;
%params.gamma = rand(4,3);
params.lambda_w = zeros(3,1) + 2.5;
params.sigma_rho_w = .05;
params.sigma_delta_w = .025;
params.sigma_gamma_w = .025;



%params.beta = zeros(4,4) + 1;

rmat = [1 1 1 1; 2 2 2 2; 3 3 3 3; 4 4 4 4];
cmat = [1 2 3 4; 1 2 3 4; 1 2 3 4; 1 2 3 4];

% load into params
params.ad.mats = struct2cell(ad);
params.ad.ids = str2double(fieldnames(ad));
params.ad.g = 3;
params.ad.bel = [ ];
params.ad.beu = [ ];
params.ad.bed = [ ];
for (jj = 1:12)
    %params.ad.el{jj} = params.ad.mats{jj}(linds);
    %params.ad.eu{jj} = params.ad.mats{jj}(uinds);
    %params.ad.ed{jj} = diag(params.ad.mats{jj});
    mmat = repmat(params.delta(:,params.ad.g),1,4) + repmat(params.gamma(:,params.ad.g)',4,1);
    params.ad.mul{jj} = mmat(linds);
    params.ad.muu{jj} = mmat(uinds);
    params.ad.mud{jj} = diag(mmat);
    params.ad.bel = [params.ad.bel params.ad.mats{jj}(linds)];
    params.ad.beu = [params.ad.beu params.ad.mats{jj}(uinds)];
    params.ad.bed = [params.ad.bed diag(params.ad.mats{jj})'];
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end

params.mci.mats = struct2cell(mci);
params.mci.ids = str2double(fieldnames(mci));
params.mci.g = 2;
params.mci.bel = [ ];
params.mci.beu = [ ];
params.mci.bed = [ ];
for (jj = 1:63)
    %params.mci.el{jj} = params.mci.mats{jj}(linds);
    %params.mci.eu{jj} = params.mci.mats{jj}(uinds);
    %params.mci.ed{jj} = diag(params.mci.mats{jj});
    mmat = repmat(params.delta(:,params.mci.g),1,4) + repmat(params.gamma(:,params.mci.g)',4,1);
    params.mci.mul{jj} = mmat(linds);
    params.mci.muu{jj} = mmat(uinds);
    params.mci.mud{jj} = diag(mmat);
    params.mci.bel = [params.mci.bel params.mci.mats{jj}(linds)];
    params.mci.beu = [params.mci.beu params.mci.mats{jj}(uinds)];
    params.mci.bed = [params.mci.bed diag(params.mci.mats{jj})'];
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end

params.nl.mats = struct2cell(nl);
params.nl.ids = str2double(fieldnames(nl));
params.nl.g = 1;
params.nl.bel = [ ];
params.nl.beu = [ ];
params.nl.bed = [ ];
for (jj = 1:37)
    %params.nl.el{jj} = params.nl.mats{jj}(linds);
    %params.nl.eu{jj} = params.nl.mats{jj}(uinds);
    %params.nl.ed{jj} = diag(params.nl.mats{jj});
    mmat = repmat(params.delta(:,params.nl.g),1,4) + repmat(params.gamma(:,params.nl.g)',4,1);
    params.nl.mul{jj} = mmat(linds);
    params.nl.muu{jj} = mmat(uinds);
    params.nl.mud{jj} = diag(mmat);
    params.nl.bel = [params.nl.bel params.nl.mats{jj}(linds)];
    params.nl.beu = [params.nl.beu params.nl.mats{jj}(uinds)];
    params.nl.bed = [params.nl.bed diag(params.nl.mats{jj})'];
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end
params.niter = niter;
params.nparms = 1 + 3 + 4 + 12 + 4 + 12 + 1 + 2 + 3;
params.uinds = uinds;
params.linds = linds;


% call the bnet_mcmc function
pvals = bnet_mcmc(params);

end
