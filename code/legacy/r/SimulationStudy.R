####################################################
##### Some Simulations to check distributions ######
####################################################
## this file was altered on 09/18/2019 by ad
## note to accompany changes:
## the pdf used in the mcmc sampling is not correct as it did not
## contain the marginal distributions.

source("~/Dropbox/Publish/BrainCGRG/code/upper_ind.R")
source("~/Dropbox/Publish/BrainCGRG/code/lower_ind.R")
# 5 by 5 network

linds = lower_ind(5);
uinds = upper_ind(5);

#  set rho
rr = 10

#  set delta
dd = c(2,-2,.25,-.5,1);

#  set gamma
gg = c(15,.5,-2,2,-1);

mus = matrix(rep(dd,5),5,5,byrow=F) + matrix(rep(gg,5),5,5,byrow=T)

mu_l = mus[linds]; mu_u = mus[uinds]

cop_pdf = function(u,v,rho=rr){
  num = rho*(1-exp(-rho))*exp(-rho*(u+v));
  denom = (1-exp(-rho) - (1-exp(-rho*u))*(1-exp(-rho*v)))^2;
  return(num/denom)
}

mat1 = mat2 = matrix(0,10000001,10)
mat1[1,] = rnorm(10,mu_l); mat2[1,] = rnorm(10,mu_u);


for (jj in c(1:10000000)){
  uu = rnorm(10,mat1[jj,])
  vv = rnorm(10,mat2[jj,])
  alpha = (prod(cop_pdf(pnorm(uu,mu_l),pnorm(vv,mu_u))) * prod(dnorm(uu,mu_l)) * prod(dnorm(vv,mu_u)))/(prod(cop_pdf(pnorm(mat1[jj,],mu_l),pnorm(mat2[jj,],mu_u)))* prod(dnorm(mat1[jj,],mu_l)) * prod(dnorm(mat2[jj,],mu_u)))
  if(runif(1) < alpha){
    mat1[(jj+1),] = uu; mat2[(jj+1),] = vv; 
  }
  else{
    mat1[(jj+1),] = mat1[jj,]; mat2[(jj+1),] = mat2[jj,]; 
  }
}
myseq = seq(from = 100000, to=10000000,by = 100000)

mylist <- vector("list",length(myseq))
for (j in c(1:length(myseq))){
  tmp <- matrix(0,5,5)
  tmp[linds] = mat1[myseq[j],]
  tmp[uinds] = mat2[myseq[j],]
  mylist[[j]] <- tmp
  names(mylist)[j] <- j
}

library(R.matlab)
writeMat(con="~/Dropbox/Publish/BrainCGRG/data/sim_list.mat", sim=mylist)

plot(mat1[myseq,1],mat2[myseq,1])
cor(mat1[myseq,1],mat2[myseq,1])

cor(cbind(mat1[myseq,],mat2[myseq,]))

#######################################################
########  Simulation #2 Multiple groups
#######################################################
source("~/Dropbox/Publish/BrainCGRG/code/upper_ind.R")
source("~/Dropbox/Publish/BrainCGRG/code/lower_ind.R")
# 5 by 5 network

linds = lower_ind(5);
uinds = upper_ind(5);

# set sd
mysd = 2;

#  set rho
rr = -7

#  set delta
dd = c(2,-2,.25,-.5,1);

#  set gamma
gg = c(15,.5,-2,2,-1);

mus = matrix(rep(dd,5),5,5,byrow=F) + matrix(rep(gg,5),5,5,byrow=T)

mu_l = mus[linds]; mu_u = mus[uinds]

cop_pdf = function(u,v,rho=rr){
  num = rho*(1-exp(-rho))*exp(-rho*(u+v));
  denom = (1-exp(-rho) - (1-exp(-rho*u))*(1-exp(-rho*v)))^2;
  return(num/denom)
}

mat1 = mat2 = matrix(0,10000001,10)
mat1[1,] = rnorm(10,mu_l); mat2[1,] = rnorm(10,mu_u);


for (jj in c(1:10000000)){
  uu = rnorm(10,mat1[jj,])
  vv = rnorm(10,mat2[jj,])
  #alpha = prod(cop_pdf(pnorm(uu,mu_l),pnorm(vv,mu_u)))/prod(cop_pdf(pnorm(mat1[jj,],mu_l),pnorm(mat2[jj,],mu_u)))  # this was the old way and is wrong
  alpha = (prod(cop_pdf(pnorm(uu,mu_l),pnorm(vv,mu_u))) * prod(dnorm(uu,mu_l)) * prod(dnorm(vv,mu_u)))/(prod(cop_pdf(pnorm(mat1[jj,],mu_l),pnorm(mat2[jj,],mu_u)))* prod(dnorm(mat1[jj,],mu_l)) * prod(dnorm(mat2[jj,],mu_u)))
  
  if(runif(1) < alpha){
    mat1[(jj+1),] = uu; mat2[(jj+1),] = vv; 
  }
  else{
    mat1[(jj+1),] = mat1[jj,]; mat2[(jj+1),] = mat2[jj,]; 
  }
}
myseq = seq(from = 100000, to=10000000,by = 50000)

mylist <- vector("list",length(myseq))
for (j in c(1:length(myseq))){
  tmp <- matrix(0,5,5)
  tmp[linds] = mat1[myseq[j],]
  tmp[uinds] = mat2[myseq[j],]
  mylist[[j]] <- tmp
  names(mylist)[j] <- j
}

library(R.matlab)
#writeMat(con="~/Dropbox/Publish/BrainCGRG/data/sim1_list.mat", sim1=mylist)
writeMat(con="~/Dropbox/Publish/BrainCGRG/data/sim1_list4paper.mat", sim1=mylist)
## 2

#  set rho
rr = -3

#  set delta
dd = c(-1,.33,8,-5,9);

#  set gamma
gg = c(3,-12,0,4,-3);

mus = matrix(rep(dd,5),5,5,byrow=F) + matrix(rep(gg,5),5,5,byrow=T)

mu_l = mus[linds]; mu_u = mus[uinds]

cop_pdf = function(u,v,rho=rr){
  num = rho*(1-exp(-rho))*exp(-rho*(u+v));
  denom = (1-exp(-rho) - (1-exp(-rho*u))*(1-exp(-rho*v)))^2;
  return(num/denom)
}

mat1 = mat2 = matrix(0,10000001,10)
mat1[1,] = rnorm(10,mu_l); mat2[1,] = rnorm(10,mu_u);


for (jj in c(1:10000000)){
  uu = rnorm(10,mat1[jj,])
  vv = rnorm(10,mat2[jj,])
  #alpha = prod(cop_pdf(pnorm(uu,mu_l),pnorm(vv,mu_u)))/prod(cop_pdf(pnorm(mat1[jj,],mu_l),pnorm(mat2[jj,],mu_u)))
  alpha = (prod(cop_pdf(pnorm(uu,mu_l),pnorm(vv,mu_u))) * prod(dnorm(uu,mu_l)) * prod(dnorm(vv,mu_u)))/(prod(cop_pdf(pnorm(mat1[jj,],mu_l),pnorm(mat2[jj,],mu_u)))* prod(dnorm(mat1[jj,],mu_l)) * prod(dnorm(mat2[jj,],mu_u)))
  
  if(runif(1) < alpha){
    mat1[(jj+1),] = uu; mat2[(jj+1),] = vv; 
  }
  else{
    mat1[(jj+1),] = mat1[jj,]; mat2[(jj+1),] = mat2[jj,]; 
  }
}
myseq = seq(from = 100000, to=10000000,by = 50000)

mylist <- vector("list",length(myseq))
for (j in c(1:length(myseq))){
  tmp <- matrix(0,5,5)
  tmp[linds] = mat1[myseq[j],]
  tmp[uinds] = mat2[myseq[j],]
  mylist[[j]] <- tmp
  names(mylist)[j] <- j
}

library(R.matlab)
#writeMat(con="~/Dropbox/Publish/BrainCGRG/data/sim2_list.mat", sim2=mylist)
writeMat(con="~/Dropbox/Publish/BrainCGRG/data/sim2_list4paper.mat", sim2=mylist)


