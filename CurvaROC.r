require(mlbench)
data("BreastCancer")
head(BreastCancer)
summary(BreastCancer)
require(skimr)
skim(BreastCancer)

BreastCancer <- na.omit(BreastCancer)
BreastCancer$Id <- NULL

set.seed(2)
ind <- sample(2, nrow(BreastCancer), replace = TRUE, prob=c(0.8, 0.2))

require(party)
x.ct <- ctree(Class ~ ., data=BreastCancer[ind == 1,])
x.ct.pred <- predict(x.ct, data=BreastCancer[ind == 2,])
x.ct.prob <-  1- unlist(treeresponse(x.ct, BreastCancer[ind == 2,]), use.names=F)[seq(1,nrow(BreastCancer[ind == 2,])*2,2)]
plot(x.ct, main="Arbol CT")

x.cf <- cforest(Class ~ ., data=BreastCancer[ind == 1,])
print(x.cf)
x.cf.pred <- predict(x.cf, newdata=BreastCancer[ind == 2,])
x.cf.prob <-  1- unlist(treeresponse(x.cf, BreastCancer[ind == 2,]), use.names=F)[seq(1,nrow(BreastCancer[ind == 2,])*2,2)]

require(ROCR)

x.ct.prob.roc <- prediction(x.ct.prob, BreastCancer[ind == 2, 'Class'])
print(x.ct.prob.roc)
x.ct.perf <- performance(x.ct.prob.roc, "tpr","fpr")
print(x.ct.perf)
plot(x.ct.perf, col=4, main="Curva ROC")

legend(0.6,0.6, c('ctree','cforest'),4:5)

x.cf.prob.roc <- prediction(x.cf.prob, BreastCancer[ind == 2,'Class'])
print(x.cf.prob.roc )
x.cf.perf <- performance(x.cf.prob.roc,"tpr","fpr")
plot(x.cf.perf, col=5, add=TRUE)
