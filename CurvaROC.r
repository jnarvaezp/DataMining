require(mlbench)
require(skimr)
data(BreastCancer)
head(BreastCancer)
skim(BreastCancer)
BreastCancer <- na.omit(BreastCancer)
BreastCancer$Id <- NULL

set.seed(2)
ind <- sample(2, nrow(BreastCancer), replace=TRUE, prob = c(0.8,0.2))

require(party)

x.ct <- ctree(Class ~ ., data=BreastCancer[ind ==1,])
x.ct.pred <- predict(x.ct, newdata=BreastCancer[ind == 2,])
x.ct.prob <- 1- unlist(treeresponse(x.ct, BreastCancer[ind == 2,]), use.names = F)[seq(1,nrow(BreastCancer[ind ==2,])*2,2)]
plot(x.ct, main="Arboles")

x.cf <- cforest(Class ~ ., data=BreastCancer[ind ==1,], control = cforest_unbiased(mtry = ncol(BreastCancer)-2))
print(x.cf)
x.cf.pred <- predict(x.ct, newdata=BreastCancer[ind == 2,])
x.cf.prob <- 1- unlist(treeresponse(x.ct, BreastCancer[ind == 2,]), use.names = F)[seq(1,nrow(BreastCancer[ind ==2,])*2,2)]

require(ROCR)

#ctree
x.ct.prob.rocr <- prediction(x.ct.prob, BreastCancer[ind == 2,'Class'])
x.ct.perf <- performance(x.ct.prob.rocr, "tpr","fpr")
# add=TRUE draws on the existing chart 
plot(x.ct.perf, col=4, main="Comparativo de curvas ROC")

# Draw a legend.
legend(0.6, 0.6, c('ctree', 'cforest'), 4:6)

# cforest
x.cf.prob.rocr <- prediction(x.cf.prob, BreastCancer[ind == 2,'Class'])
x.cf.perf <- performance(x.cf.prob.rocr, "tpr","fpr")
plot(x.cf.perf, col=5, add=TRUE)

