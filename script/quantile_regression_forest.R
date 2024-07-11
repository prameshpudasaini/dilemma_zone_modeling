library(data.table)
library(quantregForest)
library(caret)
library(brms)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load and process data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data <- fread("ignore/Wejo/trips_stop_go/model_data_FTS.txt", sep = '\t')
data$Node <- NULL
data$Approach <- NULL

# convert binary variables to factors or effect coding of binary variables
cols_factorize <- c('num_TH_lanes', 'has_shared_RT', 'has_median', 'is_weekend', 'is_night')
cols_effect_code <- cols_factorize[2:length(cols_factorize)]

# for (col in cols_factorize){
#     data[, (col) := as.factor(data[[col]])]
# }

for (col in cols_effect_code){
    data[, (col) := ifelse(data[[col]] == 0, -1, 1)]
}

# # add square of speed as variable
# data[, speed_sq := speed^2]

# specify target and predictors
X <- as.matrix(data[, !'Xi', with = FALSE])
y = data$Xi

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# feature importance using Quantile Regression Forest and cross-validation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# specify quantile value and cv folds
quantile_value <- 0.05
cv_number <- 5

# Cross-validation setup
fitControl <- trainControl(method = "cv", number = cv_number, savePredictions = TRUE, verboseIter = TRUE)

# Model tuning grid
tuneGrid <- expand.grid(mtry = c(3, 5, 7), ntree = c(100, 200, 500, 1000))

# Custom train function for quantregForest with quantiles
custom_train <- function(x, y, wts, param, lev, last, weights, ...) {
    model <- quantregForest(x, y, ntree = param$ntree, mtry = param$mtry, 
                            nodesize = 5, importance = TRUE)
    return(model)
}

# Custom predict function for quantregForest to handle quantiles
custom_predict <- function(modelFit, newdata, submodels = NULL) {
    preds <- predict(modelFit, newdata, what = quantile_value)
    return(preds)
}

# Custom function to extract variable importance
custom_varImp <- function(object, ...) {
    importance(object)
}

# Custom method definition
qrf_custom <- list(
    type = "Regression",
    library = "quantregForest",
    loop = NULL,
    parameters = data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree")),
    grid = function(x, y, len = NULL, search = "grid") { tuneGrid },
    fit = custom_train,
    predict = custom_predict,
    varImp = custom_varImp,
    prob = NULL,
    tags = c("Random Forest", "Ensemble Model", "Quantile Regression"),
    sort = function(x) x[order(x$ntree, x$mtry),],
    levels = function(x) NULL
)

# Train model using caret with custom method
qrf_model <- train(
    x = X, y = y,
    method = qrf_custom,
    trControl = fitControl,
    tuneGrid = tuneGrid,
    importance = TRUE
)

# Best model summary
print(qrf_model)

# Extract and aggregate feature importance across all folds
importance_values <- varImp(qrf_model, scale = FALSE)$importance

# create a data table of importance values and save file
importance_values <- as.data.table(importance_values, keep.rownames = 'id')
setnames(importance_values, old = 'id', new = 'feature')
# fwrite(importance_values, file = "ignore/results/QRF_importance_values_cv_5.csv")
