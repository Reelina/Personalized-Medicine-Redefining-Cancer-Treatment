######################## Modeling the clinical evidence texts and application of machine learning algorithm ###################3
######################## for classification of genetic mutation###############################



#### Calling libraries####

require(xlsx);
require(bestglm);
require(MNP);
require(foreign);
require(ggplot2);
require(MASS);
require(Hmisc);
library(kimisc)
require(reshape2);
require(car);
library(reshape);
library(aod);
library(sqldf);
library(data.table);
library(ordinal);
library(caret);

####Setting working directory######

setwd('D:/Reelina/Kaggle_Comp/Data_and_Codes/')

#### Reading train and test data files###

library("readxl")
train <-  read_excel("training_text_gene_variation_with mapping formula.xlsx")
test<- read_excel("test_text_gene_variation_with mapping formula.xlsx")

#### Creating framework for tfidf in R [else classification of classes using clinical evidence text will
#### not be possible by training the data
install.packages('tm')
library(tm)

corpus <- Corpus(VectorSource(train$Text)) # change class
summary(corpus) # get a summary
corpus <- tm_map(corpus,content_transformer(removePunctuation))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus,content_transformer(stripWhitespace))
library(SnowballC)
corpus <- tm_map(corpus, stemDocument, language="english")

inspect(corpus[1])

# Creating use a Document-Term Matrix (DTM) representation
matrix <-DocumentTermMatrix(corpus,control = list(weighting = weightTfIdf))
matrix <- removeSparseTerms(matrix, 0.95)
matrix

# first review of the matrix
inspect(matrix[1,1:100])

# A word cloud

findFreqTerms(matrix,4)

library(wordcloud)
freq = data.frame(sort(colSums(as.matrix(matrix)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=80, colors=brewer.pal(5, "Dark2"))

######### Predictive Modeling #############

# We have to remove the actual texual content for statistical model building from  thne data
train$Text = NULL
train_model = cbind(train, as.matrix(matrix))

# Preapring a training data frames for each individual class's clasification modeling

train_class1<- train_model[,c(6,15:ncol(train_model))]
train_class2<- train_model[,c(7,15:ncol(train_model))]
train_class3<- train_model[,c(8,15:ncol(train_model))]
train_class4<- train_model[,c(9,15:ncol(train_model))]
train_class5<- train_model[,c(10,15:ncol(train_model))]
train_class6<- train_model[,c(11,15:ncol(train_model))]
train_class7<- train_model[,c(12,15:ncol(train_model))]
train_class8<- train_model[,c(13,15:ncol(train_model))]
train_class9<- train_model[,c(14,15:ncol(train_model))]

#######Preparation of test set######

#install.packages('tm')
library(tm)

corpus_test <- Corpus(VectorSource(test$Text)) # change class
summary(corpus_test) # get a summary
corpus_test <- tm_map(corpus_test,content_transformer(removePunctuation))
corpus_test <- tm_map(corpus_test, content_transformer(tolower))
corpus_test <- tm_map(corpus_test, removeWords, stopwords("english"))
corpus_test <- tm_map(corpus_test,content_transformer(stripWhitespace))
corpus_test <- tm_map(corpus_test, stemDocument, language="english")

inspect(corpus_test[1])

matrix_test <-DocumentTermMatrix(corpus_test,
                                 ## without this line predict won't work
                                 control=list(dictionary=names(train_model)))

findFreqTerms(matrix_test,4)
library(wordcloud)
freq_test= data.frame(sort(colSums(as.matrix(matrix_test)), decreasing=TRUE))
wordcloud(rownames(freq_test), freq_test[,1], max.words=80, colors=brewer.pal(5, "Dark2"))

test$Text = NULL
test_model = cbind(test, as.matrix(matrix_test))

#Loading libraries respectively for predictive models- Random Forest, Logistic Regression, SVM

library(rpart)
library(rpart.plot)
library('e1071')
library(nnet)

####################################################
### Training the models for class1 and predicting class1 for Test set ####
################################################################################

class1_tree_train = rpart(class1~.,  method = "class", data = train_class1)  
prp(class1_tree_train)
class1_tree_glm_train = glm(class1~ ., family = "binomial", data =train_class1, maxit = 100)  
class1_tree_nnet_train = nnet(class1~., data=train_class1, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class1_tree_pred_test = predict(class1_tree_train,test_model,type="class")
confusionMatrix(test_model$class1,class1_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class1_glm_pred_test = as.numeric(predict(class1_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class1,class1_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class1_nnet_pred_test= as.numeric(predict(class1_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class1,class1_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")

####################################################
### Training the models for class2 and predicting class2 for Test set ####
################################################################################

class2_tree_train = rpart(class2~.,  method = "class", data = train_class2)  
prp(class2_tree_train)
class2_tree_glm_train = glm(class2~ ., family = "binomial", data =train_class2, maxit = 100)  
class2_tree_nnet_train = nnet(class2~., data=train_class2, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class2_tree_pred_test = predict(class2_tree_train,test_model,type="class")
confusionMatrix(test_model$class2,class2_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class2_glm_pred_test = as.numeric(predict(class2_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class2,class2_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class2_nnet_pred_test= as.numeric(predict(class2_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class2,class2_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")


####################################################
### Training the models for class3 and predicting class3 for Test set ####
################################################################################

class3_tree_train = rpart(class3~.,  method = "class", data = train_class3)  
prp(class3_tree_train)
class3_tree_glm_train = glm(class3~ ., family = "binomial", data =train_class3, maxit = 100)  
class3_tree_nnet_train = nnet(class3~., data=train_class3, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class3_tree_pred_test = predict(class3_tree_train,test_model,type="class")
confusionMatrix(test_model$class3,class3_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class3_glm_pred_test = as.numeric(predict(class3_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class3,class3_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class3_nnet_pred_test= as.numeric(predict(class3_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class3,class3_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")


####################################################
### Training the models for class4 and predicting class4 for Test set ####
################################################################################

class4_tree_train = rpart(class4~.,  method = "class", data = train_class4)  
prp(class4_tree_train)
class4_tree_glm_train = glm(class4~ ., family = "binomial", data =train_class4, maxit = 100)  
class4_tree_nnet_train = nnet(class4~., data=train_class4, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class4_tree_pred_test = predict(class4_tree_train,test_model,type="class")
confusionMatrix(test_model$class4,class4_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class4_glm_pred_test = as.numeric(predict(class4_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class4,class4_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class4_nnet_pred_test= as.numeric(predict(class4_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class4,class4_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")

####################################################
### Training the models for class5 and predicting class5 for Test set ####
################################################################################

class5_tree_train = rpart(class5~.,  method = "class", data = train_class5)  
prp(class5_tree_train)
class5_tree_glm_train = glm(class5~ ., family = "binomial", data =train_class5, maxit = 100)  
class5_tree_nnet_train = nnet(class5~., data=train_class5, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class5_tree_pred_test = predict(class5_tree_train,test_model,type="class")
confusionMatrix(test_model$class5,class5_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class5_glm_pred_test = as.numeric(predict(class5_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class5,class5_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class5_nnet_pred_test= as.numeric(predict(class5_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class5,class5_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")

####################################################
### Training the models for class6 and predicting class6 for Test set ####
################################################################################

class6_tree_train = rpart(class6~.,  method = "class", data = train_class6)  
prp(class6_tree_train)
class6_tree_glm_train = glm(class6~ ., family = "binomial", data =train_class6, maxit = 100)  
class6_tree_nnet_train = nnet(class6~., data=train_class6, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class6_tree_pred_test = predict(class6_tree_train,test_model,type="class")
confusionMatrix(test_model$class6,class6_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class6_glm_pred_test = as.numeric(predict(class6_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class6,class6_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class6_nnet_pred_test= as.numeric(predict(class6_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class6,class6_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")


####################################################
### Training the models for class7 and predicting class7 for Test set ####
################################################################################

class7_tree_train = rpart(class7~.,  method = "class", data = train_class7)  
prp(class7_tree_train)
class7_tree_glm_train = glm(class7~ ., family = "binomial", data =train_class7, maxit = 100)  
class7_tree_nnet_train = nnet(class7~., data=train_class7, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class7_tree_pred_test = predict(class7_tree_train,test_model,type="class")
confusionMatrix(test_model$class7,class7_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class7_glm_pred_test = as.numeric(predict(class7_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class7,class7_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class7_nnet_pred_test= as.numeric(predict(class7_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class7,class7_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")

####################################################
### Training the models for class8 and predicting class8 for Test set ####
################################################################################

class8_tree_train = rpart(class8~.,  method = "class", data = train_class8)  
prp(class8_tree_train)
class8_tree_glm_train = glm(class8~ ., family = "binomial", data =train_class8, maxit = 100)  
class8_tree_nnet_train = nnet(class8~., data=train_class8, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class8_tree_pred_test = predict(class8_tree_train,test_model,type="class")
confusionMatrix(test_model$class8,class8_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class8_glm_pred_test = as.numeric(predict(class8_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class8,class8_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class8_nnet_pred_test= as.numeric(predict(class8_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class8,class8_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")

####################################################
### Training the models for class9 and predicting class9 for Test set ####
################################################################################

class9_tree_train = rpart(class9~.,  method = "class", data = train_class9)  
prp(class9_tree_train)
class9_tree_glm_train = glm(class9~ ., family = "binomial", data =train_class9, maxit = 100)  
class9_tree_nnet_train = nnet(class9~., data=train_class9, linout=T,
                              size = 1, rang = 0.05,decay = 5e-4, maxit = 400,MaxNWts=2000)


# Getting predictions for evaluation on test sets

#Decision Tree
class9_tree_pred_test = predict(class9_tree_train,test_model,type="class")
confusionMatrix(test_model$class9,class9_tree_pred_test,dnn=c("Obs","Pred"),positive="1")


#Logistic regression
class9_glm_pred_test = as.numeric(predict(class9_tree_glm_train, test_model, type="response") > 0.40)
confusionMatrix(test_model$class9,class9_glm_pred_test,dnn=c("Obs","Pred"),positive="1")


#Neural Network
class9_nnet_pred_test= as.numeric(predict(class9_tree_nnet_train,test_model) > 0.40)
confusionMatrix(test_model$class9,class9_nnet_pred_test, dnn=c("Obs","Pred"),positive="1")

##############################################################################################################

##Bringing the predicted classification of the classes togather, for each model applied##

tree_predictions<-         data.frame(class1_tree_pred_test,
                                      class2_tree_pred_test,
                                      class3_tree_pred_test,
                                      class4_tree_pred_test,
                                      class5_tree_pred_test,
                                      class6_tree_pred_test,
                                      class7_tree_pred_test,
                                      class8_tree_pred_test,
                                      class9_tree_pred_test)

GLM_predictions<- as.data.frame(cbind(class1_glm_pred_test,
                                      class2_glm_pred_test,
                                      class3_glm_pred_test,
                                      class4_glm_pred_test,
                                      class5_glm_pred_test,
                                      class6_glm_pred_test,
                                      class7_glm_pred_test,
                                      class8_glm_pred_test,
                                      class9_glm_pred_test))

NN_predictions<- as.data.frame(cbind(class1_nnet_pred_test,
                                     class2_nnet_pred_test,
                                     class3_nnet_pred_test,
                                     class4_nnet_pred_test,
                                     class5_nnet_pred_test,
                                     class6_nnet_pred_test,
                                     class7_nnet_pred_test,
                                     class8_nnet_pred_test,
                                     class9_nnet_pred_test))

write.csv(tree_predictions,"tree_predictions.csv")
write.csv(GLM_predictions,"GLM_predictions.csv")
write.csv(NN_predictions,"NN_predictions.csv")
