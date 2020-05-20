library(ISLR)  # For OJ, and Carseats datasets
library(caret)  # for workflow
library(rpart.plot)  # for better formatted plots than the ones in rpart
library(tidyverse)
library(skimr)  # neat alternative to glance + summary
library(e1071)
library(ranger)

oj_dat <- OJ
skim_to_wide(oj_dat)


set.seed(12345)
partition <- createDataPartition(y = oj_dat$Purchase, p = 0.8, list = FALSE)
oj.train <- oj_dat[partition, ]
oj.test <- oj_dat[-partition, ]
rm(partition)

set.seed(123)
oj.full_class <- rpart(formula = Purchase ~ .,
                       data = oj.train,
                       method = "class",  # classification (not regression)
                       xval = 10  # 10-fold cross-validation 
)
rpart.plot(oj.full_class, yesno = TRUE)


printcp(oj.full_class)

plotcp(oj.full_class)

#The dashed line is set at the minimum xerror + xstd. 
#Any value below the line would be considered statistically significant. 
#A good choice for CP is often the largest value for which the error is
#within a standard deviation of the mimimum error. In this case, the smallest cp is at 0.01. 
#A good way to detect and capture the correct cp is with the which.min() function, but if 
#you want to choose the smallest statistically equivalent tree, specify it manually. Use the prune() f
#unction to prune the tree by specifying the associated cost-complexity cp.
#
oj.class <- prune(oj.full_class, 
                  cp = oj.full_class$cptable[which.min(oj.full_class$cptable[, "xerror"]), "CP"])
rm(oj.full_class)
rpart.plot(oj.class, yesno = TRUE)


oj.class.pred <- predict(oj.class, oj.test, type = "class")
plot(oj.test$Purchase, oj.class.pred, 
     main = "Simple Classification: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")


(oj.class.conf <- confusionMatrix(data = oj.class.pred, 
                                  reference = oj.test$Purchase))

oj.class.acc <- as.numeric(oj.class.conf$overall[1])
rm(oj.class.pred)
rm(oj.class.conf)


oj.class2 = train(Purchase ~ ., 
                  data = oj.train, 
                  method = "rpart",  # for classification tree
                  tuneLength = 5,  # choose up to 5 combinations of tuning parameters (cp)
                  metric='ROC',  # evaluate hyperparamter combinations with ROC
                  trControl = trainControl(
                    method = "cv",  # k-fold cross validation
                    number = 10,  # 10 folds
                    savePredictions = "final",       # save predictions for the optimal tuning parameter
                    classProbs = TRUE,  # return class probabilities in addition to predicted values
                    summaryFunction = twoClassSummary  # for binary response variable
                  )
)
oj.class2


plot(oj.class2)

oj.class.pred <- predict(oj.class2, oj.test, type = "raw")
plot(oj.test$Purchase, oj.class.pred, 
     main = "Simple Classification: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")


(oj.class.conf <- confusionMatrix(data = oj.class.pred, 
                                  reference = oj.test$Purchase))


oj.class.acc2 <- as.numeric(oj.class.conf$overall[1])
rm(oj.class.pred)
rm(oj.class.conf)
rpart.plot(oj.class2$finalModel)


plot(varImp(oj.class2), main="Variable Importance with Simple Classication")


myGrid <-  expand.grid(cp = (0:1)/10)
oj.class3 = train(Purchase ~ ., 
                  data = oj.train, 
                  method = "rpart",  # for classification tree
                  tuneGrid = myGrid,  # choose up to 5 combinations of tuning parameters (cp)
                  metric='ROC',  # evaluate hyperparamter combinations with ROC
                  trControl = trainControl(
                    method = "cv",  # k-fold cross validation
                    number = 10,  # 10 folds
                    savePredictions = "final",       # save predictions for the optimal tuning parameter
                    classProbs = TRUE,  # return class probabilities in addition to predicted values
                    summaryFunction = twoClassSummary  # for binary response variable
                  )
)
rm(myGrid)
oj.class3

plot(oj.class3)

oj.class.pred <- predict(oj.class3, oj.test, type = "raw")
plot(oj.test$Purchase, oj.class.pred, 
     main = "Simple Classification: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")


(oj.class.conf <- confusionMatrix(data = oj.class.pred, 
                                  reference = oj.test$Purchase))


oj.class.acc3 <- as.numeric(oj.class.conf$overall[1])
rm(oj.class.pred)
rm(oj.class.conf)
rpart.plot(oj.class3$finalModel)

plot(varImp(oj.class3), main="Variable Importance with Simple Classication")


rbind(data.frame(model = "Manual Class", Acc = round(oj.class.acc, 5)), 
      data.frame(model = "Caret w/tuneLength", Acc = round(oj.class.acc2, 5)),
      data.frame(model = "Caret w.tuneGrid", Acc = round(oj.class.acc3, 5))
)

#Regression Trees

carseats_dat <- Carseats
skim(carseats_dat)

partition <- createDataPartition(y = carseats_dat$Sales, p = 0.8, list = FALSE)
carseats.train <- carseats_dat[partition, ]
carseats.test <- carseats_dat[-partition, ]
rm(partition)


set.seed(1234)
# Specify model = TRUE to handle plotting splits with factor variables.
carseats.full_anova <- rpart(formula = Sales ~ .,
                             data = carseats.train,
                             method = "anova", 
                             xval = 10,
                             model = TRUE)
rpart.plot(carseats.full_anova, yesno = TRUE)

printcp(carseats.full_anova)

plotcp(carseats.full_anova)


carseats.anova <- prune(carseats.full_anova, 
                        cp = 0.039)
rpart.plot(carseats.anova, yesno = TRUE)

rm(carseats.full_anova)


carseats.anova.pred <- predict(carseats.anova, carseats.test, type = "vector")
plot(carseats.test$Sales, carseats.anova.pred, 
     main = "Simple Regression: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")
abline(0, 1)


(carseats.anova.rmse <- RMSE(pred = carseats.anova.pred,
                             obs = carseats.test$Sales))


rm(carseats.anova.pred)


carseats.anova2 = train(Sales ~ ., 
                        data = carseats.train, 
                        method = "rpart",  # for classification tree
                        tuneLength = 5,  # choose up to 5 combinations of tuning parameters (cp)
                        metric = "RMSE",  # evaluate hyperparamter combinations with RMSE
                        trControl = trainControl(
                          method = "cv",  # k-fold cross validation
                          number = 10,  # 10 folds
                          savePredictions = "final"       # save predictions for the optimal tuning parameter
                        )
)

carseats.anova2


plot(carseats.anova2)


carseats.anova.pred <- predict(carseats.anova2, carseats.test, type = "raw")
plot(carseats.test$Sales, carseats.anova.pred, 
     main = "Simple Regression: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")

(carseats.anova.rmse2 <- RMSE(pred = carseats.anova.pred,
                              obs = carseats.test$Sales))

rm(oj.class.pred)

rpart.plot(carseats.anova2$finalModel)

plot(varImp(carseats.anova2), main="Variable Importance with Simple Regression")

myGrid <-  expand.grid(cp = (0:2)/10)
carseats.anova3 = train(Sales ~ ., 
                        data = carseats.train, 
                        method = "rpart",  # for classification tree
                        tuneGrid = myGrid,  # choose up to 5 combinations of tuning parameters (cp)
                        metric = "RMSE",  # evaluate hyperparamter combinations with RMSE
                        trControl = trainControl(
                          method = "cv",  # k-fold cross validation
                          number = 10,  # 10 folds
                          savePredictions = "final"       # save predictions for the optimal tuning parameter
                        )
)
carseats.anova3

plot(carseats.anova3)
carseats.anova.pred <- predict(carseats.anova3, carseats.test, type = "raw")
plot(carseats.test$Sales, carseats.anova.pred, 
     main = "Simple Regression: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")

(carseats.anova.rmse3 <- RMSE(pred = carseats.anova.pred,
                              obs = carseats.test$Sales))

rm(carseats.anova.pred)
rpart.plot(carseats.anova3$finalModel)

plot(varImp(carseats.anova3), main="Variable Importance with Simple Regression")
rbind(data.frame(model = "Manual ANOVA", 
                 RMSE = round(carseats.anova.rmse, 5)), 
      data.frame(model = "Caret w/tuneLength", 
                 RMSE = round(carseats.anova.rmse2, 5)),
      data.frame(model = "Caret w.tuneGrid", 
                 RMSE = round(carseats.anova.rmse3, 5))
)


#Random Forests
#Random forests improve bagged trees by way of a small tweak that de-correlates the trees.
#As in bagging, the algorithm builds a number of decision trees on bootstrapped training samples. 
#But when building these decision trees, each time a split in a tree is considered, a random sample 
#of mtry predictors is chosen as split candidates from the full set of p predictors. A fresh sample 
#of mtry predictors is taken at each split. Typically mtry∼b‾√. Bagged trees are thus a special case 
#of random forests where mtry = p.

oj.bag = train(Purchase ~ ., 
               data = oj.train, 
               method = "treebag",  # for bagging
               tuneLength = 5,  # choose up to 5 combinations of tuning parameters
               metric = "ROC",  # evaluate hyperparamter combinations with ROC
               trControl = trainControl(
                  method = "cv",  # k-fold cross validation
                  number = 10,  # k=10 folds
                  savePredictions = "final",       # save predictions for the optimal tuning parameters
                  classProbs = TRUE,  # return class probabilities in addition to predicted values
                  summaryFunction = twoClassSummary  # for binary response variable
               )
)
oj.bag

#plot(oj.bag$)
oj.pred <- predict(oj.bag, oj.test, type = "raw")
plot(oj.test$Purchase, oj.pred, 
     main = "Bagging Classification: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")


(oj.conf <- confusionMatrix(data = oj.pred, 
                            reference = oj.test$Purchase))

oj.bag.acc <- as.numeric(oj.conf$overall[1])
rm(oj.pred)
rm(oj.conf)
#plot(oj.bag$, oj.bag$finalModel$y)
plot(varImp(oj.bag), main="Variable Importance with Simple Classication")

oj.frst = train(Purchase ~ ., 
                data = oj.train, 
                method = "ranger",  # for random forest
                tuneLength = 5,  # choose up to 5 combinations of tuning parameters
                metric = "ROC",  # evaluate hyperparamter combinations with ROC
                trControl = trainControl(
                   method = "cv",  # k-fold cross validation
                   number = 10,  # 10 folds
                   savePredictions = "final",       # save predictions for the optimal tuning parameter1
                   classProbs = TRUE,  # return class probabilities in addition to predicted values
                   summaryFunction = twoClassSummary  # for binary response variable
                )
)
oj.frst

plot(oj.frst)


oj.pred <- predict(oj.frst, oj.test, type = "raw")
plot(oj.test$Purchase, oj.pred, 
     main = "Random Forest Classification: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")

plot(oj.frst)

oj.pred <- predict(oj.frst, oj.test, type = "raw")
plot(oj.test$Purchase, oj.pred, 
     main = "Random Forest Classification: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")


(oj.conf <- confusionMatrix(data = oj.pred, 
                            reference = oj.test$Purchase))

oj.frst.acc <- as.numeric(oj.conf$overall[1])
rm(oj.pred)
rm(oj.conf)
#plot(oj.bag$, oj.bag$finalModel$y)
#plot(varImp(oj.frst), main="Variable Importance with Simple Classication")


rbind(data.frame(model = "Manual Class", Accuracy = round(oj.class.acc, 5)), 
      data.frame(model = "Class w/tuneLength", Accuracy = round(oj.class.acc2, 5)),
      data.frame(model = "Class w.tuneGrid", Accuracy = round(oj.class.acc3, 5)),
      data.frame(model = "Bagging", Accuracy = round(oj.bag.acc, 5)),
      data.frame(model = "Random Forest", Accuracy = round(oj.frst.acc, 5))
) %>% arrange(desc(Accuracy))
