#read the training data
install.packages("readxl")
#analyze the zeros, missing values (NA), infinity, data type, and number of unique values for a given dataset
install.packages("funModeling")
install.packages("tidyverse")
install.packages("Hmisc")
install.packages("DataExplorer")
install.packages("dataMaid")
install.packages("dlookr")
install.packages("caret")
install.packages("dplyr")
install.packages("ggplot")
install.packages("e1071")
install.packages("pROC")
install.packages("Boruta") #Feature sampling
library(Boruta)
library(pROC)
library(e1071)
library(randomForest)
library(ggplot2)
library(dplyr)
library(caret)
library(dlookr)
library(dataMaid)
library(DataExplorer)
library(Hmisc)
library(tidyverse)
library(funModeling)
library(readxl)
#read the file
radio_rf <- read_excel("RF_TrainingDatasetA_Final.xlsx")
radio_rf
View(radio_rf)
## Exploratory data analysis
#names of the variables
names(radio_rf)
#dimension of the dataset
dim(radio_rf)
#run all in one
basic_eda <- function(radio_rf)
{
  glimpse(radio_rf)
  df_status(radio_rf)
  freq(radio_rf) 
  profiling_num(radio_rf)
  plot_num(radio_rf)
  describe(radio_rf)
}
basic_eda(radio_rf)

#look into data
glimpse(radio_rf)
#Getting the metrics about data types, zeros, infinite numbers, and missing values
status(radio_rf)
#look for NA values in data
# Profiling the data input
df_status(radio_rf)
#any NA's in data
is.na(radio_rf)
#total NA's in dataset
sum(is.na(radio_rf))
colSums(is.na(radio_rf))
# Note: Majority of NA values are in outcome variable
#assign original data to another data frame
radio_rf_clean <- radio_rf
#remove the NA values using complete cases
radio_rf_clean <- na.omit(radio_rf_clean)
sum(is.na(new_data))


################### DATA PRE_PROCESSING ######################
#Removing non-zero variance predictors
nzv <- nearZeroVar(radio_rf_clean, saveMetrics= TRUE)
nzv[nzv$nzv,]
dim(radio_rf_clean)
nzv <- nearZeroVar(radio_rf_clean)
filtered_radio <- radio_rf_clean[, -nzv]
dim(filtered_radio)

#Identify correlated predictors
str(filtered_radio)
sum(is.na(filtered_radio))
#filtered_radio <- na.omit(filtered_radio)
#create a subset of data frame filtered radio
filtered_radio %>% select_if(is.numeric)->num_data
#View(num_data)
dim(num_data)
#sum(is.na(num_data))
#num_data <- na.omit(num_data)

descrCor <- cor(num_data)
summary(descrCor[upper.tri(descrCor)])

#Remove the highly correlated values from the data
highlyCorDescr <- findCorrelation(descrCor, cutoff = .85)
num_data <- num_data[,-highlyCorDescr]
descrCor2 <- cor(num_data)
summary(descrCor2[upper.tri(descrCor2)])
dim(num_data)


#Filter character variables
filtered_radio %>% select_if(is.character)->chr_data
#sum(is.na(chr_data))
#chr_data <- na.omit(chr_data)
dim(chr_data)

final_data <- data.frame(num_data,chr_data)
dim(final_data)
#sum(is.na(final_traning_data))
#View(final_traning_data)

#Remove unwanted columns from data
set.seed(183)
colnames(final_data)
final_data <- final_data[-c(30,32,33,34,35,36,37,39,40,41,42)]
colnames(final_data)
dim(final_data)

############### Data Visulisation #############
#Since Eng_class is the response variable and others are predictors
#Variable importance
variable_impo <- var_rank_info(final_traning_data,"Eng_Class")
variable_impo

# Plotting 
set.seed(183)
ggplot(variable_impo, 
       aes(x = reorder(var, gr), 
           y = gr, fill = var)
) + 
  geom_bar(stat = "identity") + 
  coord_flip() + 
  theme_bw() + 
  xlab("") + 
  ylab("Variable Importance 
       (based on Information Gain)"
  ) + 
  guides(fill = FALSE)

#Is Outcome the feature that explains the target the most?
#No, this variable was used to generate the target, thus we must exclude it. It is a typical mistake when 
#developing a predictive model to have either an input variable that was built in the same way as the 
#target (as in this case) or adding variables from the future.
#Convert Eng_Class into factor variable
final_data$Eng_Class <- as.factor(final_data$Eng_Class)
class(Eng_Class)

#Bar plot between Eng_Class and RXthresholdcriteria1
ggplot(final_data, 
       aes(x = RXthresholdcriteria1, 
           fill = Eng_Class)) + 
  geom_bar(position = "stack")

ggplot(final_data, 
       aes(x = RXthresholdcriteria2, 
           fill = Eng_Class)) + 
  geom_bar(position = "stack")


str(final_data)

#Box plot for Eng_class and Radiomodel2
ggplot(final_data, aes(x = final_data$Eng_Class, 
                     y = final_data$DpQ_R2)) +
  geom_boxplot(notch = TRUE, 
               fill = "cornflowerblue", 
               alpha = .7) +
  labs(title = "Eng class with Dpq level")

#final_data <- final_data[-c(35)]
#colnames(final_data)

#class imbalance 
barplot(prop.table(table(final_data$Eng_Class)),
        col = rainbow(2),
        ylim = c(0,0.7),
        main = "class distribution")
        


#################Question2###################

#Data Partition

library(caret)
#Random Forest
set.seed(183)
inTraining <- createDataPartition(final_data$Eng_Class, p =0.70, list = FALSE)
training <-  final_data[inTraining,]
testing <-  final_data[-inTraining,]
dim(training)
dim(testing)

colnames(training)
training <- training[-c(34)]
########## Support Vector Machine ##################
set.seed(183)
svmfit <- svm(formula = Eng_Class~., data = training, kernel = "linear", cost = 10, scale = FALSE, type = "C-classification")


#what are the support vectors??
svmfit$index

#summary of model svm
summary(svmfit)

#Observation_1: Tha result of summary is total number of support vectors are 159 when the cost=10, where 81 variables are
#from the category 1 and 78 are from category 2

#lets change the cost
svmfit_new <- svm(formula = Eng_Class~., data = training, kernel = "linear", cost = 0.1, scale = FALSE, type = "C-classification")
summary(svmfit_new)


trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
set.seed(183)
svm_Radial <- train(Eng_Class ~., data = training, method = "svmRadial",
                      trControl=trctrl,
                      preProcess = c("center", "scale"),
                      tuneLength = 10)
svm_Radial
plot(svm_Radial)

#It’s showing that final sigma parameter’s value is 0.008927611 & C parameter’s value as 8 
#Let’s try to test our model’s accuracy on our test set. 
#For predicting, we will use predict() with model’s parameters as svm_Radial & newdata= testing.

test_pred_Radial <- predict(svm_Radial, newdata = testing)
confusionMatrix(table(test_pred_Radial, testing$Eng_Class))

#Tune parameters
# grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
#                                      0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
#                            C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
#                                  1, 1.5, 2,5))
# set.seed(183)
# svm_Radial_Grid <- train(Eng_Class ~., data = training, method = "svmRadial",
#                          trControl=trctrl,
#                          preProcess = c("center", "scale"),
#                          tuneGrid = grid_radial,
#                          tuneLength = 10)



##### Model 2 KNN##########

set.seed(183)

# Create a resampling method
cv <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 5,
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary
)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(
  k = floor(seq(1, nrow(training)/3, length.out = 20))
)

# Fit knn model and perform grid search
knn_grid <- train(
  Eng_Class~., 
  data = training, 
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "ROC"
)

ggplot(knn_grid)

knnPredict <- predict(knn_grid,newdata = testing )
knnPredict
# #Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(table(knnPredict, testing$Eng_Class ))



###knn


default_knn_mod = train(
  Eng_Class ~ .,
  data = training,
  method = "knn",
  trControl = trainControl(method = "cv", number = 5)
)
default_knn_mod


# Accuracy 87 at k=9

#Tune the Hyper parameter of knn

default_knn_mod = train(
  Eng_Class ~ .,
  data = training,
  method = "knn",
  trControl = trainControl(method = "cv", number = 5),
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(k = seq(1, 101, by = 2))
)

default_knn_mod


ggplot(default_knn_mod) + theme_bw()
default_knn_mod$bestTune

#get best value of k after tuning

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}
get_best_result(default_knn_mod)

default_knn_mod$finalModel

#Predict on test data

knnPredict <- predict(default_knn_mod,newdata = testing )
knnPredict

#confusion matrix
confusionMatrix(table(knnPredict, testing$Eng_Class ))

###### Model3 Random Forest#########
# Random Search
#Resampling technique
# specify that the resampling method is 


fit_control <- trainControl(## 10-fold CV
  method = "cv",
  number = 10)

set.seed(183)
# run a random forest model

rf_fit <- train(Eng_Class ~ ., 
                data = training, 
                method = "ranger",
                trControl = fit_control)
rf_fit

# define a grid of parameter options to try
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5),
                       splitrule = c("gini", "extratrees"),
                       min.node.size = c(1, 3, 5))
rf_grid

# re-fit the model with the parameter grid
rf_fit <- train(Eng_Class ~ ., 
                data = training, 
                method = "ranger",
                trControl = fit_control,
                # provide a grid of parameters
                tuneGrid = rf_grid)
rf_fit

#Prediction on test data
rfPredict <- predict(rf_fit,newdata = testing )
rfPredict

#confusion matrix
confusionMatrix(table(rfPredict, testing$Eng_Class ))
#http://www.rebeccabarter.com/blog/2017-11-17-caret_tutorial/ (Reference)



############Question3 Feature selection##########
#Final Model == Random Forest
training$Eng_Class <- as.factor(training$Eng_Class)
str(training)
set.seed(183)
boruta <- Boruta(Eng_Class~., data = training, doTrace = 2, maxRuns = 500)
print(boruta)

#Plot
plot(boruta, las=2, cex.asis = 0.5)

#plot variable imporatnce
plotImpHistory(boruta)

attStats(boruta)

#Create random forest model with the confirmed features
#step1 Test random forest on train data with all the varaibles(34)
training[sapply(training, is.character)] <- lapply(training[sapply(training, is.character)], as.factor)
str(training)


rf34 <- randomForest(Eng_Class~., data = training)
rf34

testing$Eng_Class <- as.factor(testing$Eng_Class)
testing[sapply(testing, is.character)] <- lapply(testing[sapply(testing, is.character)], as.factor)
str(testing)



#Issue :Error in predict.randomForest(rf34, newdata = testing) : 
#Type of predictors in new data do not match that of the training data.

# Solution: his happens because your factor variables in training set and test set have different 
#levels(to be more precise test set doesn't have some of the levels present in training). 
#So you can solve this for example by using:
#levels(test$SectionName) <- levels(train$SectionName)

levels(testing$Eng_Class) <- levels(training$Eng_Class)
levels(testing$Polarization) <- levels(training$Polarization)
levels(testing$RXthresholdcriteria1) <- levels(training$RXthresholdcriteria1)
levels(testing$RXthresholdcriteria2) <- levels(training$RXthresholdcriteria2)
#levels(testing$Outcome) <- levels(training$Outcome)

#Prediction and confusion matrix
p <- predict(rf34, newdata = testing)
confusionMatrix(p, testing$Eng_Class)

# First,Build Model with NOt Important variables
#getNonRejectedFormula(boruta)


#First, Build model with Confirmed important features
getConfirmedFormula(boruta)
set.seed(183)
rf24 <- randomForest(Eng_Class ~ AntennagaindBd1 + AntennagaindBd2 + AtmosphericabsorptionlossdB + 
                       ERPdbm2 + ERPwatts1 + ERPwatts2 + FlatfademarginmultipathdB1 + 
                       FreespacelossdB + FrequencyMHz + Geoclimaticfactor + MainnetpathlossdB1 + 
                       MainreceivesignaldBm2 + OtherTXlossdB2 + Pathlengthkm + R_Powerfd1 + 
                       R_Powerfd2 + TXpowerdBm2 + DpQ_R2 + Fullmaxt1 + Fullmint1 + 
                       Polarization + RXthresholdcriteria1 + RXthresholdcriteria2, data = training
                    )
rf24



#Prediction and confusion matrix
p <- predict(rf24, newdata = testing)
confusionMatrix(p, testing$Eng_Class)

# https://www.youtube.com/watch?v=VEBax2WMbEA&t=300s (reference)

######Question4##########
#As per the probelem statement of Mr. Daniel, Where he is worried about the Radio mast outages. Being an machine
#Engineer, I build few models namely, Support vector machine with the accuracy of 98.80%, K-nearest neighbour with
#accuracy of 84.23% and lastly random forest with the accuracy of 99.80 close to 100% (Rare to be seen).

#Workflow of my Model(Random Forest)
#Step1: Random forest is the most powerful Supervised (Input and output is given) machine learning algorith.
#As the name suggests "Random Forest", it creates multiple decision tress, In general,more the number 
#of descision tress more robust the prediction and high accuracy. RF Avoids overfitting and can deal with
#large number of features.Since my target variable is categorical, Here RF will work as a "classifier"

#step2: Parameters of random forest 
#1) nTrees : number of decision tress default is 500
#2)mtry : : Number of variable/predictors(p) is randomly collected to be sampled at each split time, for classification it
#is sq.root(p) and p/3 for regression.

#Step3:Resample the traning data: Resampling is an approach to improve the accuracy of the model by applying
#various perfoemence mesaures on the data. Here, we are using Cross valiadtion: This process randomly divide
#the dataset into k folds of equal size where the first fold is trated as validation set and the model is fit
#on the remaining set, In order to perfrom the cross validation, I am using traincontrol function(caret package),
#for cross validation.

#Step4:Fit rf model using method "Ranger" stands for the random forest generator, is a fast way to implement the
#random forest model.

##Result: After performing the 10 fold cross validation on the traning data, we get the final model accuracy 100%
#with mtry = 18 and splitrule = gini

#Cost function: Cost function use to measure the performence of machine learning model.
#Gini indeax and entropy uses as a cost function in the random forest. The gini index is the measure of the 
#impurity in the model introduced bya variable.Mean decrease in the accuracy of the model due to a variable is
#determied during the out of bag error. Since the accuracy here is 100% hence the OOB error is 0%. Permuting a useful variable, 
#tend to give relatively large decrease in mean gini-gain. GINI importance is closely related to the local decision function, 
#that random forest uses to select the best available split

#Confusion Matrix: A performence matrix. As per the result of confusion matrix, it is clamied that the 543 observatios
#were correctly classified as okay and 76 correcly classied as under there was no missclassificaion of observations.
#Hence the missclassification value is 0.

############Question5###############
##Cost sensetive models:Cost-Sensitive Learning is a type of learning in 
#data mining that takes the misclassification costs
#(and possibly other types of cost) into consideration. 
#The goal of this type of learning is to minimize the total cost.

install.packages("C50")
library(C50)

cost_mat <- matrix(c(0, 1,8 ,0), nrow = 2)
rownames(cost_mat) <- colnames(cost_mat) <- c("under", "okay")
cost_mat


#Random Forest Model when h =8
model_rf<-randomForest(Eng_Class ~ ., data = training,ntree=50,nodesize=20,cp=0.005,parms = list(loss = cost_mat))
model_rf$confusion



#when h=16

cost_mat <- matrix(c(0, 1,16 ,0), nrow = 2)
rownames(cost_mat) <- colnames(cost_mat) <- c("under", "okay")
cost_mat

model_rf<-randomForest(Eng_Class ~ ., data = training,ntree=50,nodesize=20,cp=0.005,parms = list(loss = cost_mat))
model_rf$confusion

#when h=24

cost_mat <- matrix(c(0, 1,24 ,0), nrow = 2)
rownames(cost_mat) <- colnames(cost_mat) <- c("under", "okay")
cost_mat

model_rf<-randomForest(Eng_Class ~ ., data = training,ntree=50,nodesize=20,cp=0.005,parms = list(loss = cost_mat))
model_rf$confusion





#########Question6##################
scoring_data <- read_excel("RF_ScoringDatasetA_Final.xlsx")
scoring_data
View(scoring_data)

#Predict on scoring dataset
rfPredict <- predict(rf_fit,newdata = scoring_data )
rfPredict

table(rfPredict)












  








