####Objective####
#Objective: Investigate if customer responses to some survey questions (e.g. income, age, etc.) enable us to predict whether customers prefer the Acer or Sony brand.


####Set Working Directory####
setwd("C:/Users/Saad/Desktop/Data Analytics - CPE/Course 2/Task 2/C2T2") #Assigns data location


####Install and Load Packages####
install.packages("caret", dependencies = c("Depends", "Suggests")) #Installs Caret package with dependencies
library(caret) #Loads caret library


####Upload Data####
CompleteResponse <- read.csv("SurveyComplete.csv") #Reads dataset and assigns data to CompleteResponse


####Change Attribute Types (Training)####
CompleteResponse$car <- as.factor(CompleteResponse$car) #Change car attribute to factor
CompleteResponse$zipcode <- as.factor(CompleteResponse$zipcode) #Change zip code attribute to factor
CompleteResponse$brand <- as.factor(CompleteResponse$brand) #Change brand attribute to factor
CompleteResponse$brand <- factor(CompleteResponse$brand,levels = c(0,1), labels = c("Acer", "Sony"))


####Update Attribute Values (Training)####
CompleteResponse$zipcode <- factor(CompleteResponse$zipcode,levels = c(0,1,2,3,4,5,6,7,8), labels = c("New  England", "Mid-Atlantic", "East North Central", "West North Central", "South Atlantic", "East South Central", "West South Central", "Mountain", "Pacific"))
CompleteResponse$car <- factor(CompleteResponse$car,levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20), labels = c("BMW", "Buick","Cadillac","Chevrolet","Chrysler","Dodge","Ford","Honda","Hyundai","Jeep","Kia","Lincoln","Mazda","Mercedes Benz","Mitsubishi","Nissan","Ram","Subaru","Toyota","None of the above"))


#Ensure structure of dataset accurate
str(CompleteResponse)


####Data Modeling (Training)####
#Create training set equal to random 75% of original dataset. createDataPartition 
inTrain <- createDataPartition(y=CompleteResponse$brand, p=.75, list=FALSE) 


#Assign names to training and testing sets
training <- CompleteResponse[inTrain,] #Creates training set
testing <- CompleteResponse[-inTrain,] #Creates testing set
nrow(training)#Counts number of rows for training
nrow(testing) #Counts number of rows for testing
set.seed(123)#Set psuedo-random number generator

#Centers and scales data; kNN models require normalization or scaling; Random Forest does not require scaling
preProcValues <- preProcess(x=inTrain,method=c("center", "scale")) 
preProcValues #Returns scaled values


#kNN Modeling
knnctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10) #Controls computational nuances of training set through repeated cross validation (repeatedcv)
knnFit <- train(brand~., data=training, method="knn", trControl=knnctrl, preProcess=c("center", "scale")) #Creates kNN model; tuneLength value provides that many k-values (minimum value is 2)
knnFit #Returns kNN model
predictors(knnFit) #Returns all predictors
knnPredict <- predict(knnFit, testing)#Predicts kNN model to testing set
postResample(knnPredict, testing$brand) #Returns accuracy and kappa values


####Random Forest Modeling
rfctrl <- trainControl(method = "oob", number = 10) #Controls computational nuances of training set through out-of-bag estimate (oob)
rffit <- train(brand~., data=training, method="rf", trControl=rfctrl) #Creates Random Forest model 
rffit #Returns RandomForest model
predictors(rffit) #Returns all predictors
rfPredict <- predict(rffit, testing) #Predicts, using random forest model, the trained model using the testing dataset
options(max.print = 10000)
rfPredict
postResample(rfPredict, testing$brand) #Returns accuracy and kappa values


####Import Testing Dataset####
IncompleteResponse <- read.csv("SurveyIncomplete.csv") #Loads data of incomplete responses
str(IncompleteResponse)


####Change Attribute Types (Testing)####
IncompleteResponse$car <- as.factor(IncompleteResponse$car) #Change car attribute to factor
IncompleteResponse$zipcode <- as.factor(IncompleteResponse$zipcode) #Change zip code attribute to factor
IncompleteResponse$brand <- as.factor(IncompleteResponse$brand) #Change brand attribute to factor


####Change Attribute Values (Testing)####
IncompleteResponse$brand <- factor(IncompleteResponse$brand,levels = c(0,1), labels = c("Acer", "Sony"))
IncompleteResponse$zipcode <- factor(IncompleteResponse$zipcode,levels = c(0,1,2,3,4,5,6,7,8), labels = c("New  England", "Mid-Atlantic", "East North Central", "West North Central", "South Atlantic", "East South Central", "West South Central", "Mountain", "Pacific"))
IncompleteResponse$car <- factor(IncompleteResponse$car,levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20), labels = c("BMW", "Bucik","Cadillac","Chevrolet","Chrysler","Dodge","Ford","Honda","Hyundai","Jeep","Kia","Lincoln","Mazda","Mercedes Benz","Mitsubishi","Nissan","Ram","Subaru","Toyota","None of the above"))


####Data Modeling (Testing)####
#Random Forest model performed better than kNN; will use for testing
rfPredictInc <- predict(rffit, IncompleteResponse) #Random forest model as explained by Incomplete Response dataset
rfPredictInc
postResample(rfPredictInc, IncompleteResponse$brand) #Returns Accuracy and Kappa values of incomplete dataset
Conf_Matrix <- confusionMatrix(data = rfPredictInc,IncompleteResponse$brand) #Returns Acer vs Sony confusion matrix 
Conf_Matrix


####Write.csv####
output <- IncompleteResponse
output$Preference <- rfPredictInc #Assigns data from rfPredictInc into Preference attribute in output dataset
write.csv(output, file = "C2.T2Preference.csv", row.names = TRUE) #Writes in output dataset (including recently added Preference attribute to .csv file named 'C2.T2Preference.csv')
