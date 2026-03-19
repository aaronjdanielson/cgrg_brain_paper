% May 8, 2019
% 
%
%================================================================%
%        Graph Structure and Cognitive Decline                   %    
%================================================================%
%                                                                %
%          Update the mus for the three groups                   %
%                                                                %
%================================================================%


function params = bnet_update(params)
linds = params.linds;
uinds = params.uinds;

for (jj = 1:12)
    mmat = repmat(params.delta(:,params.ad.g),1,4) + repmat(params.gamma(:,params.ad.g)',4,1);
    params.ad.mul{jj} = mmat(linds);
    params.ad.muu{jj} = mmat(uinds);
    params.ad.mud{jj} = diag(mmat);
end

for (jj = 1:63)
    mmat = repmat(params.delta(:,params.mci.g),1,4) + repmat(params.gamma(:,params.mci.g)',4,1);
    params.mci.mul{jj} = mmat(linds);
    params.mci.muu{jj} = mmat(uinds);
    params.mci.mud{jj} = diag(mmat);
end

for (jj = 1:37)
    mmat = repmat(params.delta(:,params.nl.g),1,4) + repmat(params.gamma(:,params.nl.g)',4,1);
    params.nl.mul{jj} = mmat(linds);
    params.nl.muu{jj} = mmat(uinds);
    params.nl.mud{jj} = diag(mmat);
    %params.ad.mu{jj} = reshape(params.delta(rmat,params.ad.g) + params.gamma(cmat,params.ad.g),4,4);
end


end