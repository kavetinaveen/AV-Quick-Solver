#================ Title: AV - MiniHack =================#

library(h2o)

# Setting working directory
# ===========================
filepath <- c("/Users/nkaveti/Documents/Kaggle/AV_QuickSolver_MiniHack")
setwd(filepath)

# Reading data
# ==============
train <- fread("train_aPSfyYQ/train.csv")
articles <- fread("train_aPSfyYQ/article.csv")
users <- fread("train_aPSfyYQ/user.csv")
test <- fread("test.csv")
sample_sub <- fread("SampleSubmission_N8jkJEu.csv")

# 
tr_users <- unique(train$User_ID)
tr_art <- unique(train$Article_ID)

te_users <- unique(test$User_ID)
te_art <- unique(test$Article_ID)

cat("No. of new users in test data: ", length(te_users) - length(intersect(te_users, tr_users)), "\n")
cat("No. of new articles in test data: ", length(te_art) - length(intersect(te_art, tr_art)), "\n")

# Feature Engineering
train2 <- left_join(train, articles, by = "Article_ID")
train2 <- left_join(train2, users, by = "User_ID")

test2 <- left_join(test, articles, by = "Article_ID")
test2 <- left_join(test2, users, by = "User_ID")

user_rat <- train2[, .(user_mean_rat = mean(Rating), user_md_rat = median(Rating), user_max_rat = max(Rating), user_min_rat = min(Rating)), by = User_ID]
train2 <- left_join(train2, user_rat, by = "User_ID")
test2 <- left_join(test2, user_rat, by = "User_ID")

art_rat <- train2[, .(art_mean_rat = mean(Rating), art_md_rat = median(Rating), art_max_rat = max(Rating), art_min_rat = min(Rating)), by = Article_ID]
train2 <- left_join(train2, art_rat, by = "Article_ID")
test2 <- left_join(test2, art_rat, by = "Article_ID")

train2[, Article_ID := as.factor(Article_ID)]
train2[, User_ID := as.factor(User_ID)]
train2[, Var1 := as.factor(Var1)]

test2[, Article_ID := as.factor(Article_ID)]
test2[, User_ID := as.factor(User_ID)]
test2[, Var1 := as.factor(Var1)]

train2$Var1[train2$Var1 == ""] <- names(sort(table(train2$Var1), decreasing = TRUE))[1]
train2$Age[train2$Age == ""] <- names(sort(table(train2$Age), decreasing = TRUE))[2]

test2$Var1[test2$Var1 == "<NA>"] <- names(sort(table(test2$Var1), decreasing = TRUE))[1]
test2$Age[test2$Age == ""] <- names(sort(table(test2$Age), decreasing = TRUE))[2]

train2[is.na(train2)] <- 0
test2[is.na(test2)] <- 0

# Using GBM from h2o package
# --------------------------------

train_h2o <- as.h2o(train2) # Creating h2o dataframe

splits <- h2o.splitFrame(
  train_h2o,           ##  splitting the H2O frame we read above
  c(0.799,0.2),   ##  create splits of 60% and 20%; 
  ##  H2O will create one more split of 1-(sum of these parameters)
  ##  so we will get 0.6 / 0.2 / 1 - (0.6+0.2) = 0.6/0.2/0.2
  seed=1234)    ##  setting a seed will ensure reproducible results (not R's seed)

train_gbm <- h2o.assign(splits[[1]], "train.hex")
valid <- h2o.assign(splits[[2]], "valid.hex")
test_gbm <- h2o.assign(splits[[3]], "test.hex")

gbm1 <- h2o.gbm(
  training_frame = train_gbm,        ## the H2O frame for training
  validation_frame = valid,      ## the H2O frame for validation (not required)
  x=c(1,2,5:17),                        ## the predictor columns, by column index
  y=4,                          ## the target index (what we are predicting)
  model_id = "gbm_covType1",     ## name the model in H2O
  seed = 2000000)              ## Set the random seed for reproducability

summary(gbm1)                   ## View information about the model.

# test predictions
# --------------------------------
pred_test_gbm <- predict(gbm1, test_gbm)
pred_test_gbm <- data.frame(outcome = as.data.frame(test_gbm[4]), prob = as.data.frame(pred_test_gbm)[, "predict"])
rmse(pred_test_gbm[,1], pred_test_gbm[,2])

# Test predictions
# --------------------------------
Test_h2o <- as.h2o(test2)
pred_Test_gbm <- predict(gbm1, Test_h2o)
result <- as.data.frame(pred_Test_gbm)
result <- data.table(ID = test[, "ID", with = F], Rating = result$predict)
colnames(result)[1] <- "ID"
write.csv(result, file = "Predictions_GBM.csv", row.names = FALSE)









