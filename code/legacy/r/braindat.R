library(R.matlab)

fres = c()
## AD

tmp <- list.files('~/Dropbox/Matrix_Freq/AD/')

res = c()
mylist <- vector("list",12)
for (j in c(1:12)){
  mylist[[j]] <- read.table(paste('~/Dropbox/Matrix_Freq/AD/',tmp[j],sep=""),sep=",")
  tmp0 = strsplit(strsplit(tmp[j],split="S_",fixed=T)[[1]][2],split="_matrix",fixed=T)[[1]][1]
  res = c(res,as.numeric(tmp0))
  names(mylist)[j] = tmp0
}

fres = c(fres,res)

saveRDS(mylist,'~/Dropbox/Publish/BrainNetworks/data/AD_list.rds')

writeMat(con="~/Dropbox/Publish/BrainNetworks/data/AD_list.mat", AD_list=mylist)

## MCI

tmp <- list.files('~/Dropbox/Matrix_Freq/MCI/')

res = c()
mylist <- vector("list",63)
for (j in c(1:63)){
  mylist[[j]] <- read.table(paste('~/Dropbox/Matrix_Freq/MCI/',tmp[j],sep=""),sep=",")
  tmp0 = strsplit(strsplit(tmp[j],split="S_",fixed=T)[[1]][2],split="_matrix",fixed=T)[[1]][1]
  res = c(res,as.numeric(tmp0))
  names(mylist)[j] = tmp0
}

fres = c(fres,res)

saveRDS(mylist,'~/Dropbox/Publish/BrainNetworks/data/MCI_list.rds')

writeMat(con="~/Dropbox/Publish/BrainNetworks/data/MCI_list.mat", MCI_list=mylist)
## NL

tmp <- list.files('~/Dropbox/Matrix_Freq/NL/')

res = c()
mylist <- vector("list",37)
for (j in c(1:37)){
  mylist[[j]] <- read.table(paste('~/Dropbox/Matrix_Freq/NL/',tmp[j],sep=""),sep=",")
  tmp0 = strsplit(strsplit(tmp[j],split="S_",fixed=T)[[1]][2],split="_matrix",fixed=T)[[1]][1]
  res = c(res,as.numeric(tmp0))
  names(mylist)[j] = tmp0
}

fres = c(fres,res)

saveRDS(mylist,'~/Dropbox/Publish/BrainNetworks/data/NL_list.rds')

writeMat(con="~/Dropbox/Publish/BrainNetworks/data/NL_list.mat", NL_list=mylist)

saveRDS(cbind(id=fres,type=c(rep(3,12),rep(2,63),rep(1,37))),'~/Dropbox/Publish/BrainNetworks/data/scan_key.rds')

tdat = cbind(id=fres,type=c(rep(3,12),rep(2,63),rep(1,37)))

####
#dat <- read.csv('~/Dropbox/Matrix_Freq/ADNI2_SNP_rsfMRI_sample_18 (1).csv')   

#dat[,"trueID"] = sapply(as.character(dat[,"PTID"]),function(x){
#  strsplit(x,split="_S_",fixed=T)[[1]][2]
#}
#)

#length(unique(dat[,"trueID"]))

#write.csv(dat,"~/Dropbox/Publish/BrainNetworks/data/info.csv")

dat = read.csv("~/Dropbox/Publish/BrainNetworks/data/info.csv")[,-1]
sdat = subset(dat,Keep==1)
sdat[,"left"] = ifelse(sdat[,"PTHAND"]=="Left",1,0)
sdat[,"female"] = ifelse(sdat[,"PTGENDER"]=="Female",1,0)



sdat = merge(sdat[,c("trueID","left","female","PTEDUCAT","Age")],tdat,by.x="trueID",by.y="id")

sdat[,"PTEDUCAT"] = scale(sdat[,"PTEDUCAT"])
sdat[,"Age"] = scale(sdat[,"Age"])


writeMat(con="~/Dropbox/Publish/BrainNetworks/data/ptdat.mat", ptdat=as.matrix(sdat,dimnames=NULL))
#####

dat = readMat("~/Dropbox/Publish/BrainNetworks/data/ptdat.mat")$ptdat
colnames(dat) <- c("trueID","left","female","PTEDUCAT","Age","Label")
library(nnet)


test <- multinom(Label ~ left + female + PTEDUCAT + Age, data = as.data.frame(dat))

library(class)

test2 = knn(train = dat[,c("left","female","PTEDUCAT","Age")],
            test = dat[,c("left","female","PTEDUCAT","Age")],
            cl = dat[,"Label"],
            k = 5,
            prob = T)


as.numeric(as.character(apply(fitted.values(test),1,function(x){which(x==max(x))}))) - as.numeric(as.character(test2))

sum(as.numeric(as.character(apply(fitted.values(test),1,function(x){which(x==max(x))})))==dat[,"Label"])/112

sum(as.numeric(as.character(test2))==dat[,"Label"])/112

sum(rep(2,112)==dat[,"Label"])/112

knn.cv(train = dat[,c("left","female","PTEDUCAT","Age")],
         cl = dat[,"Label"],
         k = 5,
         prob = T)

dat2 = dat
dat2[,c("left","female")] = scale(dat2[,c("left","female")])

####

test3 = knn(train = dat2[,c("left","female","PTEDUCAT","Age")],
            test = dat2[,c("left","female","PTEDUCAT","Age")],
            cl = dat2[,"Label"],
            k = 5,
            prob = T)


#as.numeric(as.character(apply(fitted.values(test),1,function(x){which(x==max(x))}))) - as.numeric(as.character(test2))

#sum(as.numeric(as.character(apply(fitted.values(test),1,function(x){which(x==max(x))})))==dat[,"Label"])/112

sum(as.numeric(as.character(test3))==dat[,"Label"])/112

res = nnet(Label ~ female + left + PTEDUCAT + Age,data=dat,size = 10)

