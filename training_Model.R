library(glmnet)
library(doParallel)
require(glmnet)
require(methods)
cl <- makeCluster(5)
registerDoParallel(cl)
training_model=function(x,y){
    
 cv.lasso3 <- cv.glmnet(x, y, family='binomial', alpha=0, parallel=TRUE, standardize=FALSE, penalty.factor=penalty2,type.measure='auc')
 save(cv.lasso3,file='cv.lasso3_full_22k_new.RData')

}
