% July 4, 2019
% 
%
%================================================================%
%        Graph Structure and Cognitive Decline                   %    
%================================================================%
%                                                                %
%           Posterior Classification for CGRG                    %
%                                                                %
%================================================================%

% here we compute the posterior probabilities of class membership
% how much does the CGRG model improve our ability to classify
% units as AD, MCI and NL?
ad = load('~/Dropbox/Publish/BrainNetworks/data/AD_list.mat');
nl = load('~/Dropbox/Publish/BrainNetworks/data/NL_list.mat');

function [pvals, params] = bnet_classify(ad,mci,nl,niter)

% load Alzheimer's and Normal Cognitive Function Data
%load('~/Dropbox/Publish/BrainNetworks/data/AD_list.mat');
%load('~/Dropbox/Publish/BrainNetworks/data/NL_list.mat');

% load posterior sample

load('~/Dropbox/Publish/BrainCGRG/output/modres.mat','mypars','params');
niter = 300000;
burnin = 100000;
mypars2 = mypars((burnin+1):niter,:);
thin = 10;
nrows = numel(mypars2(:,1));
tmp = linspace(1, nrows, nrows);
tmp = tmp(1 : thin : end);  
mypars2 = mypars2(tmp,:);

% evaluate each of the networks at the three different settings
% save this as a 3d array
% number of samples x number of brains x number of groups

% First the ad group
% need to evaluate the probability of each network separately
params.ad.ids;  

% make containers to store the results for each person in the two groups
% these will have nrows equal to the number in each group and will give
% the unnormalized probability of each outcome.

NN = length(tmp);

ad_probs = zeros(NN,12,3);
nl_probs = zeros(NN,37,3);

for jj = 1:NN
    
    res = bnet_individualprob(params,mypars2(jj,:));
    for kk = 1:12
        %ww = prod(res.adad(  ((kk-1)*6 + 1): ((kk-1)*6 + 6))) + 
        ad_probs(jj,kk,1) =  prod(res.adnl(  ((kk-1)*6 + 1): ((kk-1)*6 + 6)));
        ad_probs(jj,kk,2) =  prod(res.admci(  ((kk-1)*6 + 1): ((kk-1)*6 + 6)));
        ad_probs(jj,kk,3) =  prod(res.adad(  ((kk-1)*6 + 1): ((kk-1)*6 + 6)));
        %ad_prob0(jj,kk,1) =  ad_prob0(jj,kk,1)/
    end
  
    for kk = 1:37
        nl_probs(jj,kk,1) =  prod(res.nlnl(  ((kk-1)*6 + 1): ((kk-1)*6 + 6)));
        nl_probs(jj,kk,2) =  prod(res.nlmci(  ((kk-1)*6 + 1): ((kk-1)*6 + 6)));
        nl_probs(jj,kk,3) =  prod(res.nlad(  ((kk-1)*6 + 1): ((kk-1)*6 + 6)));
    end
    
end

save('~/Dropbox/Publish/BrainCGRG/output/probability_matrices_ad_nl.mat','ad_probs','nl_probs')

% now combine these with the multinomial 
% sort multinomial so it fits with the order of ids per group.
% multiply each of the numbers in ad_probs and nl_probs by the
% corresponding multinomial term.  Then sum and divide

%mult = load('~/Dropbox/Publish/BrainCGRG/output/class_probs_mult.mat','res');
load('~/Dropbox/Publish/BrainCGRG/output/mult_ad.mat','mult_ad')
load('~/Dropbox/Publish/BrainCGRG/output/mult_nl.mat','mult_nl')

ad_prob0 = zeros(NN,12,3);
nl_prob0 = zeros(NN,37,3);


for jj = 1:NN
    for kk = 1:12
        %ww = prod(res.adad(  ((kk-1)*6 + 1): ((kk-1)*6 + 6))) + 
        tmp1 =  ad_probs(jj,kk,1) * mult_ad(kk,1);
        tmp2 =  ad_probs(jj,kk,2) * mult_ad(kk,2);
        tmp3 =  ad_probs(jj,kk,3) * mult_ad(kk,3);
        ww = tmp1 + tmp2 + tmp3;
        ad_prob0(jj,kk,1) =  tmp1/ww;
        ad_prob0(jj,kk,2) =  tmp2/ww;
        ad_prob0(jj,kk,3) =  tmp3/ww;
        %ad_prob0(jj,kk,1) =  ad_prob0(jj,kk,1)/
    end
  
    for kk = 1:37
        nl_prob0(jj,kk,1) =  nl_probs(jj,kk,1) * mult_nl(kk,1);
        nl_prob0(jj,kk,2) =  nl_probs(jj,kk,2) * mult_nl(kk,2);
        nl_prob0(jj,kk,3) =  nl_probs(jj,kk,3) * mult_nl(kk,3);
        ww = nl_prob0(jj,kk,1) + nl_prob0(jj,kk,2) + nl_prob0(jj,kk,3);
        nl_prob0(jj,kk,1) =  nl_prob0(jj,kk,1)/ww;
        nl_prob0(jj,kk,2) =  nl_prob0(jj,kk,2)/ww;
        nl_prob0(jj,kk,3) =  nl_prob0(jj,kk,3)/ww;
    end
    
end

save('~/Dropbox/Publish/BrainCGRG/output/final_probability_matrices_ad_nl.mat','ad_prob0','nl_prob0')


scatterplot



end