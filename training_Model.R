
library(glmnet)
library(doParallel)
require(glmnet)
require(methods)
library(pROC)
require(caret)
cl <- makeCluster(5)
registerDoParallel(cl)
#training the model ridge regression
ridge_regression_model=function(xtrain,ytrain,xtest,ytest){
    
 cv.lasso3 <- cv.glmnet(x, y, family='binomial', alpha=0, parallel=TRUE, standardize=FALSE, penalty.factor=penalty2,type.measure='auc')
 save(cv.lasso3,file='cv.lasso3.RData')

#validation of the model
coeff=coef(cv.lasso3)
coeff=coeff[-1]

#polygenic score on validation data
PRS=as.matrix(xtest)%*% as.vector(coeff))
#Print ROC
print(roc(ytest,PRS)$auc)

}
#training the model elastic net
elastic_net_model=function(xtrain,ytrain,xtest,ytest){
ytrain=factor(ytrain)
levels(ytrain) <- make.names(levels(factor(ytrain)))
cctrl1 <- trainControl(method="cv", number=10, returnResamp="all",
                       classProbs=TRUE, summaryFunction=twoClassSummary)
test_class_cv_model <- train(x, y, method = "glmnet", 
                             trControl = cctrl1,metric = "AUC",
                             tuneGrid = expand.grid(alpha = seq(0.00001,0.2,by = 0.02),
                                                    lambda = seq(0.00001,0.16,by = 0.03)))
save(test_class_cv_model,file='elastic.RData')
plot(test_class_cv_model)
#validation of the model
my.glmnet.model <- test_class_cv_model$finalModel
#polygenic score on validation data
coeff=predict(test_class_cv_model$finalModel, type = "coefficients") 
PRS=as.matrix(xtest)%*% as.vector(coeff))
#Print ROC
print(roc(ytest,PRS)$auc)
}
