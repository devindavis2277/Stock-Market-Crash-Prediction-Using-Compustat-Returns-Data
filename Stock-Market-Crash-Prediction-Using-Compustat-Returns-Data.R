library(haven)
library(dplyr)
library(lubridate)
library(caret)
library(randomForest)
library(pROC)
library(ggplot2)

# Load Data 
mret <- read_sas("~/Desktop/mret7023.sas7bdat")

# Handle Date 
if ("DATE" %in% names(mret)) {
  if (is.numeric(mret$DATE)) {
    mret$DATE <- as.Date(mret$DATE, origin = "1960-01-01")
  } else {
    mret$DATE <- as.Date(mret$DATE)
  }
} else {
  stop("❌ Could not find a DATE column in the dataset.")
}

# Derived Predictors
mret <- mret %>%
  mutate(
    turnover = VOL / SHROUT,
    marketcap = abs(PRC) * SHROUT
  )

# Filter last 5 years 
last_date <- max(mret$DATE, na.rm = TRUE)
mret <- mret %>%
  filter(DATE >= last_date %m-% years(5)) %>%
  arrange(DATE)

# Create CRASH variable
mret <- mret %>%
  mutate(CRASH = ifelse(RET < -0.08, 1, 0))

cat("Crash class distribution:\n")
print(table(mret$CRASH))

# Lag predictors
mret <- mret %>%
  group_by(PERMNO) %>%
  arrange(DATE, .by_group = TRUE) %>%
  mutate(
    lag1_ret = lag(RET, 1),
    lag2_ret = lag(RET, 2),
    lag3_ret = lag(RET, 3),
    lag1_vol = lag(abs(RET), 1),
    lag1_turnover = lag(turnover, 1),
    lag1_mktcap = lag(marketcap, 1)
  ) %>%
  ungroup()

# Remove NA values 
mret_model <- na.omit(mret)

# Ensure both classes exist 
if (length(unique(mret_model$CRASH)) < 2) {
  stop("❌ Only one CRASH class present — try expanding your time window or lowering the crash threshold.")
}

# Train/Test Split with balance check 
set.seed(123)
trainIndex <- createDataPartition(mret_model$CRASH, p = 0.8, list = FALSE)
train <- mret_model[trainIndex, ]
test  <- mret_model[-trainIndex, ]

cat("\nTrain class counts:\n")
print(table(train$CRASH))
cat("\nTest class counts:\n")
print(table(test$CRASH))

# Safety check — re-sample if one set lacks class balance
if (length(unique(train$CRASH)) < 2 || length(unique(test$CRASH)) < 2) {
  cat("\n⚠️ Re-balancing dataset since one class is missing...\n")
  # Use stratified sampling to maintain both classes
  set.seed(42)
  trainIndex <- createDataPartition(mret_model$CRASH, p = 0.8, list = FALSE)
  train <- mret_model[trainIndex, ]
  test  <- mret_model[-trainIndex, ]
}


# Logistic Regression Model
logit_model <- glm(
  CRASH ~ lag1_ret + lag2_ret + lag3_ret + lag1_vol +
    lag1_turnover + lag1_mktcap,
  data = train, family = binomial
)

cat("\n===== Logistic Regression Summary =====\n")
print(summary(logit_model))

# Predictions
logit_probs <- predict(logit_model, newdata = test, type = "response")
logit_pred  <- ifelse(logit_probs > 0.5, 1, 0)

# Confusion Matrix (only if both classes exist)
if (length(unique(test$CRASH)) == 2) {
  print(confusionMatrix(as.factor(logit_pred), as.factor(test$CRASH)))
}

# ROC Curve & AUC
if (length(unique(test$CRASH)) == 2) {
  roc_logit <- roc(test$CRASH, logit_probs)
  cat("\nLogistic Regression AUC:", auc(roc_logit), "\n")
  plot(roc_logit, main = "ROC Curve - Logistic Regression", col = "blue", lwd = 2)
}

# Random Forest Model
rf_model <- randomForest(
  as.factor(CRASH) ~ lag1_ret + lag2_ret + lag3_ret +
    lag1_vol + lag1_turnover + lag1_mktcap,
  data = train,
  ntree = 200, mtry = 3, importance = TRUE
)

cat("\n===== Random Forest Summary =====\n")
print(rf_model)

rf_probs <- predict(rf_model, newdata = test, type = "prob")[, 2]
rf_pred  <- ifelse(rf_probs > 0.5, 1, 0)

if (length(unique(test$CRASH)) == 2) {
  print(confusionMatrix(as.factor(rf_pred), as.factor(test$CRASH)))
  
  roc_rf <- roc(test$CRASH, rf_probs)
  cat("\nRandom Forest AUC:", auc(roc_rf), "\n")
  plot(roc_rf, main = "ROC Curve - Random Forest", col = "darkgreen", lwd = 2)
}

cat("\n==== Random Forest Variable Importance =====\n")
print(importance(rf_model))
varImpPlot(rf_model, main = "Random Forest Variable Importance")



