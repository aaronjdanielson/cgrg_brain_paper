% May 7, 2019
%================================================================%
%================================================================%
%        Graph Structure and Cognitive Decline                   %    
%================================================================%
%                                                                %
%        Execution Function                                      %
%                                                                %
%================================================================%

%================================================================%

% FYI:  This file is used for code execution and software development only!!!!! 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% notes:  improve the variance structure to make it more realistic
%         create diagnostic plots
%         assess how well the model indicates differences between disease
%         states
%         add covariates - just a general set at first - not one for each
%         dyad - are the covariates indicative of greater communication in
%         the brain?  as is, there may be too many parameters to make this
%         useful. 
%         Is the model for self-ties reasonable?  It may not be.
%         In this most recent version, I'm ignoring the self-ties.

function res = execute_bnet(niter)

niter = 300000;
% load the lists containing the networks
load('~/Dropbox/Publish/BrainNetworks/data/AD_list.mat');
load('~/Dropbox/Publish/BrainNetworks/data/MCI_list.mat');
load('~/Dropbox/Publish/BrainNetworks/data/NL_list.mat');
% load the patient's covariates
load('~/Dropbox/Publish/BrainNetworks/data/ptdat.mat');

%ad = struct2cell(AD_list,MCI_list,NL_list,100);
[mypars, params] = bnet_driver(AD_list,MCI_list,NL_list,niter);

save('~/Dropbox/Publish/BrainCGRG/output/modres.mat','mypars','params')

burnin = 100000;
mypars2 = mypars((burnin+1):niter,:);
thin = 10;
nrows = numel(mypars2(:,1));
tmp = linspace(1, nrows, nrows);
tmp = tmp(1 : thin : end);  % => 1 4 7 10
mypars2 = mypars2(tmp,:);

histogram(mypars2(:,1));
histogram(mypars2(:,2));
histogram(mypars2(:,3));


[f0,xi0] = ksdensity(mypars2(:,1)); 
figure
plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,2)); 
figure
plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,3)); 
figure
plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,4)); 
figure
plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Rho (Reciprocity)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% second node's sender effect: 6, 10, 14, 18

inds = [6 10 14 18];

[f0,xi0] = ksdensity(mypars2(:,inds(1))); 
%figure
%plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,inds(2))); 
%figure
%plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,inds(3))); 
%figure
%plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,inds(4))); 
%figure
%plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Sender Effect (Second Node)');
axis([-.5 .5 0 11])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% third node's sender effect: 7, 11, 14, 18

inds = [7 11 15 19];

[f0,xi0] = ksdensity(mypars2(:,inds(1))); 
%figure
%plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,inds(2))); 
%figure
%plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,inds(3))); 
%figure
%plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,inds(4))); 
%figure
%plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Sender Effect (Second Node)');
axis([-.5 .5 0 11])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fourth node's sender effect: 7, 11, 14, 18

inds = [8 12 16 20];

[f0,xi0] = ksdensity(mypars2(:,inds(1))); 
%figure
%plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,inds(2))); 
%figure
%plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,inds(3))); 
%figure
%plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,inds(4))); 
%figure
%plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Sender Effect (Fourth Node)');
axis([-.5 .5 0 11])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first node's receiver effect: 21, 25, 29, 33

inds = [21 25 29 33];

[f0,xi0] = ksdensity(mypars2(:,inds(1))); 
%figure
%plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,inds(2))); 
%figure
%plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,inds(3))); 
%figure
%plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,inds(4))); 
%figure
%plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Receiver Effect (First Node)');
axis([-.5 .5 0 15])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% second node's receiver effect: 22, 26, 30, 34

inds = [22 26 30 34];

[f0,xi0] = ksdensity(mypars2(:,inds(1))); 
%figure
%plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,inds(2))); 
%figure
%plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,inds(3))); 
%figure
%plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,inds(4))); 
%figure
%plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Receiver Effect (Second Node)');
axis([-.5 .5 0 15])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% third node's receiver effect: 23, 27, 31, 35

inds = [23 27 31 35];

[f0,xi0] = ksdensity(mypars2(:,inds(1))); 
%figure
%plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,inds(2))); 
%figure
%plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,inds(3))); 
%figure
%plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,inds(4))); 
%figure
%plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Receiver Effect (Third Node)');
axis([-.4 .4 0 15])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fourth node's receiver effect: 24, 28, 32, 36

inds = [24 28 32 36];

[f0,xi0] = ksdensity(mypars2(:,inds(1))); 
%figure
%plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,inds(2))); 
%figure
%plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,inds(3))); 
%figure
%plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,inds(4))); 
%figure
%plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Receiver Effect (Fourth Node)');
axis([-.25 .75 0 15])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% observation error: 37, 38, 39, 40

inds = [37 38 39 40];

[f0,xi0] = ksdensity(mypars2(:,inds(1))); 
%figure
%plot(xi0,f0);

[f1,xi1] = ksdensity(mypars2(:,inds(2))); 
%figure
%plot(xi1,f1);

[f2,xi2] = ksdensity(mypars2(:,inds(3))); 
%figure
%plot(xi,f);

[f3,xi3] = ksdensity(mypars2(:,inds(4))); 
%figure
%plot(xi,f);

plot(xi0, f0, 'y', 'LineWidth', 1.2)
hold on 
X_plot = [xi0, fliplr(xi0)];
Y_plot= [zeros(100,1)', fliplr(f0)];
h1 = fill(X_plot, Y_plot , 1,....
        'facecolor','y', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi1, f1, 'm', 'LineWidth', 1.2)
X_plot = [xi1, fliplr(xi1)];
Y_plot= [zeros(100,1)', fliplr(f1)];
h2 = fill(X_plot, Y_plot , 1,....
        'facecolor','m', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi2, f2, 'r', 'LineWidth', 1.2)
X_plot = [xi2, fliplr(xi2)];
Y_plot= [zeros(100,1)', fliplr(f2)];
h3 = fill(X_plot, Y_plot , 1,....
        'facecolor','r', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
plot(xi3, f3, 'c', 'LineWidth', 1.2)
X_plot = [xi3, fliplr(xi3)];
Y_plot= [zeros(100,1)', fliplr(f3)];
h4 = fill(X_plot, Y_plot , 1,....
        'facecolor','c', ...
        'edgecolor','none', ...
        'facealpha', 0.3);

% now add the points and this will be great
%plot(x(1:17,1),x(1:17,4),"o");
%plot(xf,yf,"o");
%plot(xfh,yfh,"o");
hold off
legend
legend([h1 h2 h3 h4],'Prior','Normal','MCI','AD','Location', 'NorthEast');
%chleg = get(hleg,'children');
%set(chleg(1),'color','y');
%set(chleg(2),'color','m');
%set(chleg(3),'color','r');
%set(chleg(4),'color','c');
title('Posterior Distribution of Observation Error');
axis([0 9 0 2])


end
