##Load Libraries
library(glmnet)
library(doParallel)
require(methods)
library(pROC)
require(caret)
library(data.table)
library(ggplot2)
cl <- makeCluster(5)
registerDoParallel(cl)

#training the model ridge regression
ridge_regression_model=function(xtrain,ytrain,xtest){
  
    cv.lasso3 <- cv.glmnet(data.matrix(xtrain),data.matrix(ytrain), family='binomial', alpha=0, parallel=TRUE, standardize=FALSE, type.measure='auc')
    save(cv.lasso3,file='cv.lasso3.RData')
  
    #validation of the model
    coeff=coef(cv.lasso3)
    coeff=coeff[-1]
  
    #polygenic score on validation data
    PRS=as.matrix(xtest)%*% as.vector(coeff)
return(PRS)
}

#training the model elastic net
elastic_net_model=function(xtrain,ytrain,xtest){
    ytrain=factor(ytrain$output)
    levels(ytrain) <- make.names(levels(factor(ytrain)))
    cctrl1 <- trainControl(method="cv", number=10, returnResamp="all",
                         classProbs=TRUE, summaryFunction=twoClassSummary)
    test_class_cv_model <- train(xtrain, ytrain, method = "glmnet", 
                               trControl = cctrl1,metric = "ROC",
                               tuneGrid = expand.grid(alpha = seq(0.00001,0.2,by = 0.02),
                                                      lambda = seq(0.00001,0.16,by = 0.03)))
    save(test_class_cv_model,file='elastic.RData')
    pdf('model.pdf')
    plot(test_class_cv_model)
    dev.off()
  #validation of the model
    coeff=coef(test_class_cv_model$finalModel,test_class_cv_model$bestTune$lambda ) 
  #polygenic score on validation data
    PRS=as.matrix(xtest)%*% as.vector(coeff[-1])
return(PRS)
}

###Plot ROC
ROC=function(PRS){
    roc.list=roc(output ~ PRS1+PRS2,data=PRS)
    pdf('roc.pdf')
    print(ggroc(roc.list,size=2, font.x = c("14", "bold"),font.y = c("14", "bold")))
    dev.off()
}

############Input Data
####1. Training Data
    xtrain=fread('x_train.csv',stringsAsFactors=F,sep=',',header=T)
    ytrain=xtrain[,1]
    xtrain=xtrain[,-1]
####2. Test Data
    xtest=fread('x_test.csv',stringsAsFactors=F,sep=',',header=T)
    ytest=xtest[,1]
    xtest=xtest[,-1]

####Calling Ridge Regression
    PRS1=ridge_regression_model(xtrain,ytrain,xtest)
####Calling elastic Net
    PRS2=elastic_net_model(xtrain,ytrain,xtest)
##Plot ROC
    PRS=data.frame(PRS1,PRS2,ytest)
    ROC(PRS)





