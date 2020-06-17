library(ggplot2)
library(purrr)
library(tidyr)
library(dplyr)
library(MASS)
library(randomForest)
library(caret)
library(e1071)
library(glmnet)
library(plotly)

#Create testing and training set
set.seed(123)
training.samples <- cars$MSRP %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data <- cars[training.samples,]
test.data <- cars[-training.samples,]

# Predictor variables
x <- model.matrix(MSRP ~ Engine.HP + city.mpg + highway.MPG + Popularity,
                    train.data)[,-1]
# Outcome variable
y <- train.data$MSRP

glmnet(x, y, alpha = 1, lambda = NULL)


#ridge
set.seed(123)
cv <- cv.glmnet(x, y, alpha = 0)
# Display the best lambda value
cv$lambda.min

# Fit the final model on the training data
model <- glmnet(x, y, alpha = 0, lambda = cv$lambda.min)
# Display regression coefficients
coef(model)

# Make predictions on the test data
x.test <- model.matrix(MSRP ~ Engine.HP + city.mpg + highway.MPG + Popularity, test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$MSRP),
  Rsquare = R2(predictions, test.data$MSRP)
)

# Lasso 

# Find the best lambda using cross-validation
set.seed(123) 
cvl <- cv.glmnet(x, y, alpha = 1)
# Display the best lambda value
cvl$lambda.min

# Fit the final model on the training data
model <- glmnet(x, y, alpha = 1, lambda = cvl$lambda.min)
# Dsiplay regression coefficients
coef(model)

# Make predictions on the test data
x.test <- model.matrix(MSRP ~ Engine.HP + city.mpg + highway.MPG + Popularity, test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$MSRP),
  Rsquare = R2(predictions, test.data$MSRP)
)

#Elastic Net

# Build the model using the training set
set.seed(123)
model <- train(
  MSRP ~ Engine.HP + city.mpg + highway.MPG + Popularity, data = train.data, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Best tuning parameter
model$bestTune

# Coefficient of the final model. You need
# to specify the best lambda
coef(model$finalModel, model$bestTune$lambda)


# Make predictions on the test data
x.test <- model.matrix(MSRP ~ Engine.HP + city.mpg + highway.MPG + Popularity, test.data)[,-1]
predictions <- model %>% predict(x.test)
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$MSRP),
  Rsquare = R2(predictions, test.data$MSRP)
)

# Linear Regression
cor(cars$MSRP, cars$Popularity)
cor(cars$MSRP, cars$Engine.Cylinders)
cor(cars$MSRP, cars$highway.MPG)
cor(cars$MSRP, cars$city.mpg)
cor(cars$MSRP, cars$Engine.HP)
cor(cars$Engine.HP, cars$Engine.Cylinders)

lm1 <- lm(formula = cars$Engine.HP ~ cars$MSRP, data = cars)
lm1
summary(lm1)


set.seed(100)
trainingrowidex <- sample(1:nrow(cars), 0.8*nrow(cars))
trainingdata <- cars[trainingrowidex, ]
testdata <- cars[-trainingrowidex, ]

lmmod <- lm(MSRP ~ Engine.HP , data = trainingdata)
pred <- predict(lmmod, testdata)
summary(lmmod)
AIC(lmmod)

actualpreds <- data.frame(cbind(actauls = testdata$MSRP, predicteds = pred))
correlationaccuracy <- cor(actualpreds)
head(actualpreds)

minmaxaccuracy <- mean(apply(actualpreds, 1, min) / apply(actualpreds, 1, max))
mape <- mean(abs((actualpreds$predicteds - actualpreds$actauls)) / actualpreds$actauls)

minmaxaccuracy
mape


# Random Forest
names(cars) <- make.names(names(cars))

set.seed(100)
train <- sample(nrow(cars), 0.7*nrow(cars), replace = FALSE)
training <- cars[train,]
valid <- cars[-train,]


set.seed(100)
cars.rf <- randomForest(MSRP ~ Engine.HP + highway.MPG + city.mpg + Popularity, data = training, importance = TRUE)
cars.rf

predvalid <- predict(cars.rf, valid)
totaltest <- cbind(valid, predvalid)
reg <- lm(MSRP ~ predvalid, data = totaltest)
reg
RMSE.valid <- sqrt(mean(reg$sesiduals^2))
RMSE.valid
