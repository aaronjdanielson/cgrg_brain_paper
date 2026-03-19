% May 8, 2019
% 
%
%================================================================%
%        Graph Structure and Cognitive Decline                   %    
%================================================================%
%                                                                %
%           MCMC function for Brain CGRG                         %
%                                                                %
%================================================================%


function pvals = bnet_mcmc(params0)


rho0_w = params0.rho0_w;
rho_w = params0.rho_w;
delta0_w = params0.delta0_w;
delta_w = params0.delta_w;
gamma0_w = params0.gamma0_w;
gamma_w = params0.gamma_w;
lambda_w = params0.lambda_w;
sigma_rho_w = params0.sigma_rho_w;
sigma_delta_w = params0.sigma_delta_w;
sigma_gamma_w = params0.sigma_gamma_w;

nparms = params0.nparms;
niter = params0.niter;
linds = params0.linds;
uinds = params0.uinds;

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Storage for posterior realizations
pvals = zeros(niter,nparms);

% Start MCMC ...
fprintf('First Calculation...\n');
% Evaluate logpost at params0
lpost0 = bnet_logpost(params0);
fprintf('Looping...\n');

for iter = 1:niter
    
    % Update rho general
    
    rho01 = params0.rho0 + (rand(1)-.5)*rho0_w;
    % Evaluate logpost at new value
    params1 = params0;
    params1.rho0 = rho01;
    lpost1 = bnet_logpost(params1);
    % M-H acceptance step
    if (log(rand(1)) < lpost1 - lpost0)
        lpost0 = lpost1;
        params0 = params1;
    end
    
    % Update rho by group
    
    for k = 1:3
        rho1_k = params0.rho(k) + (rand(1)-.5)*rho_w(k);
        % Evaluate logpost at new value
        params1 = params0;
        params1.rho(k) = rho1_k;
        lpost1 = bnet_logpost(params1);
        % M-H acceptance step
        if (log(rand(1)) < lpost1 - lpost0)
            lpost0 = lpost1;
            params0 = params1;
        end
    end
    
    % Update beta general
    
    %beta1 = params0.beta + (rand(1)-.5)*rho_w;
    % Evaluate logpost at new value
    %params1 = params0;
    %params1.rho = rho1;
    %lpost1 = bnet_logpost(params1);
    % M-H acceptance step
    %if (log(rand(1)) < lpost1 - lpost0)
    %    lpost0 = lpost1;
    %    params0 = params1;
    %end
    
    % Update beta by group
    
    
    
    % Update delta
    
    for k = 2:4
        delta01_k = params0.delta0(k) + (rand(1)-.5)*delta0_w(k);
        % Evaluate logpost at new value
        params1 = params0;
        params1.delta0(k) = delta01_k;
        lpost1 = bnet_logpost(params1);
        % M-H acceptance step
        if (log(rand(1)) < lpost1 - lpost0)
            lpost0 = lpost1;
            params0 = params1;
        end
    end
    
    % Update delta by group
    
    for j = 1:3
        for k = 2:4
            %if(k ~= 1)
                delta_kj = params0.delta(k,j) + (rand(1)-.5)*delta_w(k,j);
                % Evaluate logpost at new value
                params1 = params0;
                params1.delta(k,j) = delta_kj;
                params1 = bnet_update(params1);
                lpost1 = bnet_logpost(params1);
                % M-H acceptance step
                if (log(rand(1)) < lpost1 - lpost0)
                    lpost0 = lpost1;
                    params0 = params1;
                end
            %end
        end
    end
    
    % Update gamma
    
    for k = 1:4
        gamma01_k = params0.gamma0(k) + (rand(1)-.5)*gamma0_w(k);
        % Evaluate logpost at new value
        params1 = params0;
        params1.gamma0(k) = gamma01_k;
        lpost1 = bnet_logpost(params1);
        % M-H acceptance step
        if (log(rand(1)) < lpost1 - lpost0)
            lpost0 = lpost1;
            params0 = params1;
        end
    end
    
    % Update gamma by group
    
    for j = 1:3
        for k = 1:4
            gamma1_kj = params0.gamma(k,j) + (rand(1)-.5)*gamma_w(k,j);
            % Evaluate logpost at new value
            params1 = params0;
            params1.gamma(k,j) = gamma1_kj;
            params1 = bnet_update(params1);
            lpost1 = bnet_logpost(params1);
            % M-H acceptance step
            if (log(rand(1)) < lpost1 - lpost0)
                lpost0 = lpost1;
                params0 = params1;
            end
        end
    end
    
    % Update lambda
    
    for j = 1:3
        lambda1 = params0.lambda(j) + (rand(1)-.5)*lambda_w(j);
        if lambda1 > 0
        % Evaluate logpost at new value
            params1 = params0;
            params1.lambda(j) = lambda1;
            lpost1 = bnet_logpost(params1);
            % M-H acceptance step
            if (log(rand(1)) < lpost1 - lpost0)
                lpost0 = lpost1;
                params0 = params1;
            end
        end
    end
    
    % Update sigma_rho
    
    sigma_rho1 = params0.sigma_rho + (rand(1)-.5)*sigma_rho_w;
    % Evaluate logpost at new value
    if sigma_rho1 >0
        params1 = params0;
        params1.sigma_rho = sigma_rho1;
        lpost1 = bnet_logpost(params1);
        % M-H acceptance step
        if (log(rand(1)) < lpost1 - lpost0)
            lpost0 = lpost1;
            params0 = params1;
        end
    end
    
    % Update sigma_delta
    
    sigma_delta1 = params0.sigma_delta + (rand(1)-.5)*sigma_delta_w;
    % Evaluate logpost at new value
    if sigma_delta1 > 0
        params1 = params0;
        params1.sigma_delta = sigma_delta1;
        lpost1 = bnet_logpost(params1);
        % M-H acceptance step
        if (log(rand(1)) < lpost1 - lpost0)
            lpost0 = lpost1;
            params0 = params1;
        end
    end
    
    % Update sigma_gamma
    
    sigma_gamma1 = params0.sigma_gamma + (rand(1)-.5)*sigma_gamma_w;
    % Evaluate logpost at new value
    if sigma_gamma1 > 0
        params1 = params0;
        params1.sigma_gamma = sigma_gamma1;
        lpost1 = bnet_logpost(params1);
        % M-H acceptance step
        if (log(rand(1)) < lpost1 - lpost0)
            lpost0 = lpost1;
            params0 = params1;
        end
    end
    
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    pvals(iter,:) = [params0.rho0 params0.rho' params0.delta0' reshape(params0.delta,1,12) params0.gamma0' reshape(params0.gamma,1,12) params0.lambda' params0.sigma_rho params0.sigma_delta params0.sigma_gamma]; 
    
    % Display iteration results on screen
    fprintf('%4d: ',iter);
    fprintf(' %4.4f',pvals(iter,:));
    fprintf('\n');
end

end