% May 8, 2019
% 
%
%================================================================%
%        Graph Structure and Cognitive Decline                   %    
%================================================================%
%                                                                %
%        Probability function for Brain CGRG                     %
%                                                                %
%================================================================%
% This function is used to get the probabilities of individual 
% brain networks evaluated under the three different group labels

function res = bnet_individualprob(params,pval)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% need to identify where each one is in the result from the mcmc run:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho = pval(2:4);
delta = reshape(pval(9:20),4,3);
gamma = reshape(pval(25:36),4,3);
lambda = pval(37:39);
sigma_rho = pval(40);
sigma_delta = pval(41);
sigma_gamma = pval(42);



% log-likelihood
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX%
% SPECIFY LOG - LIKELIHOOD HERE      %
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX%
% need to formulate the log-likelihood and then aggregate over all the
% observations in the three groups.


% upper and lower triangular entries for ad
% don't need to repeat this
%params.ad.bmul = [ ];
%params.ad.bmuu = [ ];
%params.ad.bmud = [ ];
%for (jj = 1:params.nad)
%    params.ad.bmul = [params.ad.bmul params.ad.mul{jj}];
%    params.ad.bmuu = [params.ad.bmuu params.ad.muu{jj}];
%    params.ad.bmud = [params.ad.bmud params.ad.mud{jj}'];
%end

linds = params.linds;
uinds = params.uinds;

% this is where the new parameters go as we loop through the sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ad as ad
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for (jj = 1:12)
    mmat = repmat(delta(:,params.ad.g),1,4) + repmat(gamma(:,params.ad.g)',4,1);
    ad.ad.mul{jj} = mmat(linds);
    ad.ad.muu{jj} = mmat(uinds);
    ad.ad.mud{jj} = diag(mmat);
end

ad.ad.bmul = [ ];
ad.ad.bmuu = [ ];
ad.ad.bmud = [ ];
for (jj = 1:params.nad)
    ad.ad.bmul = [ad.ad.bmul ad.ad.mul{jj}];
    ad.ad.bmuu = [ad.ad.bmuu ad.ad.muu{jj}];
    ad.ad.bmud = [ad.ad.bmud ad.ad.mud{jj}'];
end


adadpl = normcdf(params.ad.bel,ad.ad.bmul,1/sqrt(lambda(3)));
adadpu = normcdf(params.ad.beu,ad.ad.bmuu,1/sqrt(lambda(3)));

adadcop = bnet_copprob(adadpl,adadpu,rho(3)) .* normpdf(params.ad.bel,ad.ad.bmul,1/sqrt(lambda(3))) .* normpdf(params.ad.beu,ad.ad.bmuu,1/sqrt(lambda(3)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ad as mci
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for (jj = 1:12)
    mmat = repmat(delta(:,params.mci.g),1,4) + repmat(gamma(:,params.mci.g)',4,1);
    ad.mci.mul{jj} = mmat(linds);
    ad.mci.muu{jj} = mmat(uinds);
    ad.mci.mud{jj} = diag(mmat);
end

ad.mci.bmul = [ ];
ad.mci.bmuu = [ ];
ad.mci.bmud = [ ];
for (jj = 1:params.nad)
    ad.mci.bmul = [ad.mci.bmul ad.mci.mul{jj}];
    ad.mci.bmuu = [ad.mci.bmuu ad.mci.muu{jj}];
    ad.mci.bmud = [ad.mci.bmud ad.mci.mud{jj}'];
end

admcipl = normcdf(params.ad.bel,ad.mci.bmul,1/sqrt(lambda(2)));
admcipu = normcdf(params.ad.beu,ad.mci.bmuu,1/sqrt(lambda(2)));

admcicop = bnet_copprob(admcipl,admcipu,rho(2)) .* normpdf(params.ad.bel,ad.mci.bmul,1/sqrt(lambda(2))) .* normpdf(params.ad.beu,ad.mci.bmuu,1/sqrt(lambda(2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ad as nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for (jj = 1:12)
    mmat = repmat(delta(:,params.nl.g),1,4) + repmat(gamma(:,params.nl.g)',4,1);
    ad.nl.mul{jj} = mmat(linds);
    ad.nl.muu{jj} = mmat(uinds);
    ad.nl.mud{jj} = diag(mmat);
end

ad.nl.bmul = [ ];
ad.nl.bmuu = [ ];
ad.nl.bmud = [ ];
for (jj = 1:params.nad)
    ad.nl.bmul = [ad.nl.bmul ad.nl.mul{jj}];
    ad.nl.bmuu = [ad.nl.bmuu ad.nl.muu{jj}];
    ad.nl.bmud = [ad.nl.bmud ad.nl.mud{jj}'];
end

adnlpl = normcdf(params.ad.bel,ad.nl.bmul,1/sqrt(lambda(1)));
adnlpu = normcdf(params.ad.beu,ad.nl.bmuu,1/sqrt(lambda(1)));

adnlcop = bnet_copprob(adnlpl,adnlpu,rho(1)) .* normpdf(params.ad.bel,ad.nl.bmul,1/sqrt(lambda(1))) .* normpdf(params.ad.beu,ad.nl.bmuu,1/sqrt(lambda(1)));



% diagonal entries for ad

%laddiag = sum(log(normpdf(params.ad.bed,params.ad.bmud,1/sqrt(params.lambda(3)))));
%laddiag = 0;

% upper and lower triangular entries for mci
% we ignore mci for the classification exercise ...

% upper and lower triangular entries for nl
% do not need to repeat this, but save for now
%params.nl.bmul = [ ];
%params.nl.bmuu = [ ];
%params.nl.bmud = [ ];
%for (jj = 1:params.nnl)
%    params.nl.bmul = [params.nl.bmul params.nl.mul{jj}];
%    params.nl.bmuu = [params.nl.bmuu params.nl.muu{jj}];
%    params.nl.bmud = [params.nl.bmud params.nl.mud{jj}'];
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nl as nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for (jj = 1:37)
    mmat = repmat(delta(:,params.nl.g),1,4) + repmat(gamma(:,params.nl.g)',4,1);
    nl.nl.mul{jj} = mmat(linds);
    nl.nl.muu{jj} = mmat(uinds);
    nl.nl.mud{jj} = diag(mmat);
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end

nl.nl.bmul = [ ];
nl.nl.bmuu = [ ];
nl.nl.bmud = [ ];
for (jj = 1:params.nnl)
    nl.nl.bmul = [nl.nl.bmul nl.nl.mul{jj}];
    nl.nl.bmuu = [nl.nl.bmuu nl.nl.muu{jj}];
    nl.nl.bmud = [nl.nl.bmud nl.nl.mud{jj}'];
end

nlnlpl = normcdf(params.nl.bel,nl.nl.bmul,1/sqrt(lambda(1)));
nlnlpu = normcdf(params.nl.beu,nl.nl.bmuu,1/sqrt(lambda(1)));

nlnlcop = bnet_copprob(nlnlpl,nlnlpu,rho(1)) .* normpdf(params.nl.bel,nl.nl.bmul,1/sqrt(lambda(1))) .* normpdf(params.nl.beu,nl.nl.bmuu,1/sqrt(lambda(1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nl as ad
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for (jj = 1:37)
    mmat = repmat(delta(:,params.ad.g),1,4) + repmat(gamma(:,params.ad.g)',4,1);
    nl.ad.mul{jj} = mmat(linds);
    nl.ad.muu{jj} = mmat(uinds);
    nl.ad.mud{jj} = diag(mmat);
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end

nl.ad.bmul = [ ];
nl.ad.bmuu = [ ];
nl.ad.bmud = [ ];
for (jj = 1:params.nnl)
    nl.ad.bmul = [nl.ad.bmul nl.ad.mul{jj}];
    nl.ad.bmuu = [nl.ad.bmuu nl.ad.muu{jj}];
    nl.ad.bmud = [nl.ad.bmud nl.ad.mud{jj}'];
end

nladpl = normcdf(params.nl.bel,nl.ad.bmul,1/sqrt(lambda(3)));
nladpu = normcdf(params.nl.beu,nl.ad.bmuu,1/sqrt(lambda(3)));

nladcop = bnet_copprob(nladpl,nladpu,rho(3)) .* normpdf(params.nl.bel,nl.ad.bmul,1/sqrt(lambda(3))) .* normpdf(params.nl.beu,nl.ad.bmuu,1/sqrt(lambda(3)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nl as mci
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for (jj = 1:37)
    mmat = repmat(delta(:,params.mci.g),1,4) + repmat(gamma(:,params.mci.g)',4,1);
    nl.mci.mul{jj} = mmat(linds);
    nl.mci.muu{jj} = mmat(uinds);
    nl.mci.mud{jj} = diag(mmat);
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end

nl.mci.bmul = [ ];
nl.mci.bmuu = [ ];
nl.mci.bmud = [ ];
for (jj = 1:params.nnl)
    nl.mci.bmul = [nl.mci.bmul nl.mci.mul{jj}];
    nl.mci.bmuu = [nl.mci.bmuu nl.mci.muu{jj}];
    nl.mci.bmud = [nl.mci.bmud nl.mci.mud{jj}'];
end


nlmcipl = normcdf(params.nl.bel,nl.mci.bmul,1/sqrt(lambda(2)));
nlmcipu = normcdf(params.nl.beu,nl.mci.bmuu,1/sqrt(lambda(2)));

nlmcicop = bnet_copprob(nlmcipl,nlmcipu,rho(2)) .* normpdf(params.nl.bel,nl.mci.bmul,1/sqrt(lambda(2))) .* normpdf(params.nl.beu,nl.mci.bmuu,1/sqrt(lambda(2)));


% diagonal entries for nl

%lnldiag = sum(log(normpdf(params.nl.bed,params.nl.bmud,1/sqrt(params.lambda(1)))));
%lnldiag = 0;


%loglik = ladcop + laddiag + lmcicop + lmcidiag + lnlcop + lnldiag;

res.adad = adadcop;
res.admci = admcicop;
res.adnl = adnlcop; 
%[adadcop admcicop adnlcop];
res.nlnl = nlnlcop;
res.nlad = nladcop;
res.nlmci = nlmcicop;
%res.nl = [nlnlcop nladcop nlmcicop];

%lpost = loglik + logprior;
end