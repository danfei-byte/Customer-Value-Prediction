setwd("/Users/dynamite/Desktop/DA/Projects/ML_H&M/Source Data")
data <- read.csv("final_data.csv")
colSums(is.na(data)) # check missing values

suppressPackageStartupMessages({
  library(tidyverse)
  library(randomForest)
  library(Metrics)
})

# Convert to factors
data$club_member_status <- as.factor(data$club_member_status)
data$fashion_news_frequency <- as.factor(data$fashion_news_frequency)

# ---- Train/test split (same as your code) ----
set.seed(746)
train_ind  <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
data_train <- data[train_ind, ]
data_test  <- data[-train_ind, ]

# ---- Define formula ----
rf_formula <- aov ~ num_purchases + club_member_status + fashion_news_frequency + avg_age +
  Shoes...Socks + Blouses..Tops...Shirts + Hoodies...Outerwear +
  Skirts..Shorts...Tights + Swimwear + Trousers + Knitwear +
  T.shirts + Nightwear + Underwear + Accessories + Dresses..Jumpsuits...Sets

# ---- Choose number of trees (set 60) ----
N_TREES <- 60 

set.seed(746)
rf_model <- randomForest(
  formula   = rf_formula,
  data      = data_train,
  importance= TRUE,
  na.action = na.omit,
  ntree     = N_TREES
)

# Quick diagnostics (optional)
print(rf_model) # Mean of squared residuals: 0.0001165533; % Var explained: 25.29

# ---- Evaluate on test ----
pred_rf <- predict(rf_model, data_test)
rmse_rf <- rmse(data_test$aov, pred_rf)
r2_rf   <- caret::R2(pred_rf, data_test$aov)
cat(sprintf("Test RMSE: %.8f\nTest R^2: %.6f\n", rmse_rf, r2_rf)) # Test RMSE: 0.01076448; Test R^2: 0.246408

# ---- Build artifacts for scoring ----
# 1) Feature column names in the exact order randomForest will use
terms_obj   <- terms(rf_formula)
feature_cols <- attr(terms_obj, "term.labels")

# 2) Factor levels to avoid new-level crashes at score time
levels_bundle <- list(
  club_member_status     = levels(data_train$club_member_status),
  fashion_news_frequency = levels(data_train$fashion_news_frequency)
)

# 3) Default modes for factors (used when new levels appear / missing)
mode_of <- function(x) { names(sort(table(x), decreasing = TRUE))[1] }
default_modes <- list(
  club_member_status     = mode_of(data_train$club_member_status),
  fashion_news_frequency = mode_of(data_train$fashion_news_frequency)
)

# 4) AOV tier thresholds (40th/80th percentiles of TRAIN predictions)
train_preds <- predict(rf_model, data_train)
qs <- as.numeric(quantile(train_preds, probs = c(0.40, 0.80), na.rm = TRUE))
names(qs) <- c("p40", "p80")

# ---- Save artifacts ----
dir.create("models", showWarnings = FALSE)
saveRDS(
  list(
    model         = rf_model,
    feature_cols  = feature_cols,
    factor_levels = levels_bundle,
    default_modes = default_modes,
    n_trees       = N_TREES,
    formula       = rf_formula
  ),
  file = sprintf("models/rf_bundle_ntree_%d.rds", N_TREES)
)
saveRDS(qs, file = sprintf("models/aov_tiers_ntree_%d.rds", N_TREES))

cat(sprintf("Saved: models/rf_bundle_ntree_%d.rds\n", N_TREES))
cat(sprintf("Saved: models/aov_tiers_ntree_%d.rds (p40=%.6f, p80=%.6f)\n", N_TREES, qs["p40"], qs["p80"]))
#  models/aov_tiers_ntree_60.rds (p40=0.022335, p80=0.031260)

# src/score.R, score
install.packages("optparse")
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
})

option_list <- list(
  make_option(c("--bundle"), type = "character", default = "models/rf_bundle_ntree_60.rds",
              help = "Path to model bundle RDS"),
  make_option(c("--tiers"),  type = "character", default = "models/aov_tiers_ntree_60.rds",
              help = "Path to tiers RDS"),
  make_option(c("--input"),  type = "character", default = "final_data.csv", help = "Path to scoring CSV"),
  make_option(c("--out"),    type = "character", default = "out/aov_scores.csv",
              help = "Output CSV path")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)

# ---- Load artifacts ----
bundle <- readRDS(opt$bundle)
tiers  <- readRDS(opt$tiers)

rf_model     <- bundle$model
feature_cols <- bundle$feature_cols
lvl          <- bundle$factor_levels
modes        <- bundle$default_modes

# ---- Load scoring data ----
# Require 'customer_id' plus the exact feature columns used in training.
newdat <- read.csv(opt$input, check.names = FALSE)  # keep original names if needed
if (!"customer_id" %in% names(newdat)) stop("Input must include 'customer_id' column.")

# ---- Ensure required columns exist (add missing numeric columns as 0) ----
missing_feats <- setdiff(feature_cols, names(newdat))
if (length(missing_feats) > 0) {
  message("INFO: Adding missing numeric features as 0: ", paste(missing_feats, collapse = ", "))
  for (cn in missing_feats) newdat[[cn]] <- 0
}

# ---- Coerce factor columns to training levels & fill unknowns with mode ----
# These are your only factor predictors in the formula:
fact_cols <- c("club_member_status", "fashion_news_frequency")
for (fc in fact_cols) {
  if (!fc %in% names(newdat)) {
    # If completely missing, create with default mode
    newdat[[fc]] <- modes[[fc]]
  }
  newdat[[fc]] <- factor(newdat[[fc]], levels = lvl[[fc]])
  # Replace NAs (unseen/blank) with mode
  nas <- is.na(newdat[[fc]])
  if (any(nas)) newdat[[fc]][nas] <- modes[[fc]]
  # Re-assert levels after replacement
  newdat[[fc]] <- factor(newdat[[fc]], levels = lvl[[fc]])
}

# ---- Reorder/select predictors exactly as training expected ----
X <- newdat[, feature_cols, drop = FALSE]

# ---- Predict ----
aov_pred <- as.numeric(predict(rf_model, newdata = X))

# ---- Tiering ----
tierize <- function(x, p40, p80) {
  dplyr::case_when(
    x < p40 ~ "Low",
    x < p80 ~ "Mid",
    TRUE    ~ "High"
  )
}
aov_tier <- tierize(aov_pred, tiers[["p40"]], tiers[["p80"]])

# ---- Write output ----
out <- tibble(
  customer_id = newdat$customer_id,
  aov_pred    = aov_pred,
  aov_tier    = aov_tier
)
dir.create("out", showWarnings = FALSE)
write.csv(out, opt$out, row.names = FALSE)
message("Wrote: ", opt$out)