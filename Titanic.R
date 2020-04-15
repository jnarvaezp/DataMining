path <- 'https://raw.githubusercontent.com/jnarvaezp/DataMining/master/titanic_data.csv'
titanic <- read.csv(path)

library('ggplot2')
# Calcular la distribucion por clase  en el barco y sexo
ggplot(titanic,aes(x=factor(pclass),fill=factor(sex)))+
  geom_bar(position = "dodge")
# Calcular la distribucion por superviviencia y sexo
ggplot(titanic,aes(x=factor(pclass),fill=factor(sex)))+
  geom_bar(position = "dodge")+
  facet_grid(". ~ survived")
# Calcular y agrupar la supervivencia, sexo, edad y clase

posn.j <- position_jitter(0.5,0)
ggplot(titanic,aes(x=factor(pclass),y=age,col=factor(sex)))+
  geom_jitter(size=3,alpha=0.5,position = posn.j)+
  facet_grid(". ~ survived")
  
#create_train_test <- function(data, size = 0.8, train = TRUE) {
#  n_row <- nrow(data)
#  total_row <- size * n_row
#  train_sample <- 1:total_row
#  if (train == TRUE) {
#    return (data[train_sample, ])
#  } else {
#    return (data[-train_sample, ])
#  }
#}

#data_train <- create_train_test(clean_titanic, 0.8, train = TRUE)
#data_test <- create_train_test(clean_titanic, 0.8, train = FALSE)

# Importar el dataset de entrenamiento
train_url <- "https://raw.githubusercontent.com/jnarvaezp/DataMining/master/train.csv"
train <- read.csv(train_url)

# Importar el dataset de pruebas
test_url <- "https://raw.githubusercontent.com/jnarvaezp/DataMining/master/test.csv"
test <- read.csv(test_url)
# Print train and test to the console
#train
#test

str(train)
str(test)
# Your train and test set are still loaded
# ver los datasets

#en el dataset de entrenamiento ver la distribucion de sobrevivientes
table(train$Survived)
#en el dataset de entrenamiento ver la distribucion de sobrevivientes por genero
table(train$Sex,train$Survived)

#Calcular la proporcion respeco al sexo y los sobrevientes
prop.table(table(train$Sex,train$Survived),1)
#extraer las personas menores de 18 años
train$Child <- NA
train$Child[train$Age < 18] <- 1
train$Child[train$Age >= 18] <- 0
# Calcular la proporcion de estos datos
prop.table(table(train$Child,train$Survived),1)
#Construir el arbol
library('rpart')

my_tree_two <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class")
plot(my_tree_two)
text(my_tree_two)
#install.packages('rattle')
#install.packages('rpart.plot')
#install.packages('RColorBrewer')

library('rattle')
library('rpart.plot')
library('RColorBrewer')
#graficar el arbol

fancyRpartPlot(my_tree_two)

# Hacer predicciones basado en el test de prueba
my_prediction <- predict(my_tree_two,test,type="class")

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)
# Finish the data.frame() call

# Use nrow() on my_solution
nrow(my_solution)
ncol(my_solution)


# Finish the write.csv() call
#write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)


# Visualize my_tree_three
my_tree_three <-  rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class", control = rpart.control(minsplit = 50, cp =0))
fancyRpartPlot(my_tree_three)
#Hasta Aca
