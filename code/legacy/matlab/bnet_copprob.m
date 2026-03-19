% May 10, 2019
% 
%
%================================================================%
%        Graph Structure and Cognitive Decline                   %    
%================================================================%
%                                                                %
%           Copula Probability function                          %
%                                                                %
%================================================================%

function prob = bnet_copprob(pl,pu,rho)


num = rho*((1-exp(-rho))*exp(-rho*(pl + pu)));

denom = (1-exp(-rho) - ((1-exp(-rho*pl)).*(1-exp(-rho*pu)))).^2;
prob = num./denom;
%prob = denom;
end
