## ========================================================
## LABORATORIO 7 - Regresión Logística
## SmartStay Advisors - Optimización de precios en Airbnb
## COMMIT FINAL: Actividades 1 a 12 (modelo binario + multiclase + comparación)
## ========================================================

library(caret)
library(car)
library(ggplot2)
library(pROC)
library(glmnet)
library(profvis)
library(htmlwidgets)

# Librerías nuevas para actividades 11 y 12
library(nnet)            # multinom
library(rpart)           # árbol de decisión
library(randomForest)    # random forest
library(e1071)           # naive bayes
library(class)           # knn
library(microbenchmark)  # timing opcional

# ====================== ACTIVIDAD 1 ======================
# --- Carga y limpieza de datos --------------------------
load("listings.RData")

listings$price <- as.numeric(gsub("[$,]", "", listings$price))

if (!"precio_cat" %in% names(listings)) {
  q33 <- quantile(listings$price, 0.33, na.rm = TRUE)
  q66 <- quantile(listings$price, 0.66, na.rm = TRUE)
  listings$precio_cat <- cut(
    listings$price,
    breaks = c(-Inf, q33, q66, Inf),
    labels = c("económico", "medio", "caro"),
    include.lowest = TRUE
  )
}

listings$is_economico <- ifelse(listings$precio_cat == "económico", 1, 0)
listings$is_medio     <- ifelse(listings$precio_cat == "medio",     1, 0)
listings$is_caro      <- ifelse(listings$precio_cat == "caro",      1, 0)

cat("\nDistribución de precio_cat:\n")
print(table(listings$precio_cat))

# ====================== ACTIVIDAD 2 ======================
set.seed(123)
index <- createDataPartition(listings$precio_cat, p = 0.7, list = FALSE)
trainData <- listings[index, ]
testData  <- listings[-index, ]

predictors <- c("room_type", "accommodates", "bedrooms", "bathrooms",
                "review_scores_rating", "number_of_reviews",
                "host_is_superhost", "instant_bookable",
                "neighbourhood_cleansed", "property_type")

vars_modelo <- c("is_caro", predictors)
trainData <- trainData[complete.cases(trainData[, vars_modelo]), ]
testData  <- testData[complete.cases(testData[,  vars_modelo]), ]

testData <- testData[
  testData$neighbourhood_cleansed %in% levels(factor(trainData$neighbourhood_cleansed)) &
    testData$property_type %in% levels(factor(trainData$property_type)) &
    testData$room_type %in% levels(factor(trainData$room_type)), ]

cat("\nFilas finales - Train:", nrow(trainData), " | Test:", nrow(testData), "\n")

# ====================== ACTIVIDAD 3 ======================
trainData$caro_factor <- factor(trainData$is_caro, levels = c(0, 1), labels = c("No", "Si"))
testData$caro_factor  <- factor(testData$is_caro,  levels = c(0, 1), labels = c("No", "Si"))

trainControl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

set.seed(123)
model_caro <- train(
  caro_factor ~ .,
  data      = trainData[, c("caro_factor", predictors)],
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = trainControl,
  metric    = "ROC"
)
print(model_caro)

# ====================== ACTIVIDAD 4 ======================
final_glm <- model_caro$finalModel

cat("\n=== Coeficientes significativos (p < 0.05) ===\n")
coef_table <- summary(final_glm)$coefficients
print(head(coef_table[coef_table[, 4] < 0.05, ], 15))

num_vars <- c("accommodates", "bedrooms", "bathrooms",
              "review_scores_rating", "number_of_reviews")
cor_matrix <- cor(trainData[, num_vars], use = "complete.obs")
cat("\n=== Matriz de correlación ===\n")
print(round(cor_matrix, 2))

pseudo_r2 <- 1 - (final_glm$deviance / final_glm$null.deviance)
cat("\nPseudo R² (McFadden) =", round(pseudo_r2, 4), "\n")

# ====================== ACTIVIDAD 5 ======================
pred_probs <- predict(model_caro, newdata = testData, type = "prob")[, "Si"]
pred_class <- factor(ifelse(pred_probs > 0.5, "Si", "No"), levels = c("No", "Si"))
conf_matrix <- confusionMatrix(pred_class, testData$caro_factor, positive = "Si")
print(conf_matrix)

# ====================== ACTIVIDAD 6 ======================
train_probs <- predict(model_caro, newdata = trainData, type = "prob")[, "Si"]
train_class <- factor(ifelse(train_probs > 0.5, "Si", "No"), levels = c("No", "Si"))
train_error <- 1 - mean(train_class == trainData$caro_factor)
test_error  <- 1 - conf_matrix$overall["Accuracy"]
cat("\nError entrenamiento:", round(train_error, 4),
    " | Error prueba:", round(test_error, 4), "\n")

# Curva de aprendizaje
set.seed(123)
train_sizes  <- round(seq(0.2, 1, by = 0.1) * nrow(trainData))
train_errors <- numeric(length(train_sizes))
test_errors  <- numeric(length(train_sizes))
for (i in seq_along(train_sizes)) {
  sub_idx   <- sample(nrow(trainData), train_sizes[i])
  sub_train <- trainData[sub_idx, ]
  sub_model <- glm(caro_factor ~ .,
                   data   = sub_train[, c("caro_factor", predictors)],
                   family = binomial)
  tr_p <- predict(sub_model, sub_train, type = "response") > 0.5
  train_errors[i] <- mean(tr_p != (sub_train$caro_factor == "Si"))
  te_p <- predict(sub_model, testData, type = "response") > 0.5
  test_errors[i]  <- mean(te_p != (testData$caro_factor == "Si"))
}
learning_df <- data.frame(size = train_sizes,
                          train_error = train_errors,
                          test_error  = test_errors)
p_learn <- ggplot(learning_df, aes(x = size)) +
  geom_line(aes(y = train_error, color = "Entrenamiento"), linewidth = 1) +
  geom_line(aes(y = test_error,  color = "Prueba"),        linewidth = 1) +
  geom_point(aes(y = train_error), color = "blue") +
  geom_point(aes(y = test_error),  color = "red") +
  labs(title = "Curva de Aprendizaje - Regresión Logística (Caro / No)",
       x = "Tamaño del conjunto de entrenamiento",
       y = "Error de clasificación", color = "") +
  theme_minimal() +
  scale_color_manual(values = c("Entrenamiento" = "blue", "Prueba" = "red"))
print(p_learn)

## ========================================================
## ACTIVIDAD 7 - Tuneo del modelo (Regularización)
## ========================================================
tuneGrid <- expand.grid(
  alpha  = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10^seq(-4, 0, length.out = 15)
)

set.seed(123)
model_tuned <- train(
  caro_factor ~ .,
  data      = trainData[, c("caro_factor", predictors)],
  method    = "glmnet",
  family    = "binomial",
  trControl = trainControl,
  tuneGrid  = tuneGrid,
  metric    = "ROC"
)
cat("\n=== Mejor combinación ===\n"); print(model_tuned$bestTune)

pred_probs_tuned <- predict(model_tuned, newdata = testData, type = "prob")[, "Si"]
pred_class_tuned <- factor(ifelse(pred_probs_tuned > 0.5, "Si", "No"),
                           levels = c("No", "Si"))
conf_tuned <- confusionMatrix(pred_class_tuned, testData$caro_factor, positive = "Si")
print(conf_tuned)

## ========================================================
## ACTIVIDAD 8 - Matriz de confusión + profiler
## ========================================================
prof_base <- profvis({
  pp_b <- predict(model_caro, newdata = testData, type = "prob")[, "Si"]
  pc_b <- factor(ifelse(pp_b > 0.5, "Si", "No"), levels = c("No", "Si"))
  cm_b <- confusionMatrix(pc_b, testData$caro_factor, positive = "Si")
})
htmlwidgets::saveWidget(prof_base, "profiler_base.html", selfcontained = TRUE)

t_base <- system.time(predict(model_caro,  newdata = testData, type = "prob"))
t_tune <- system.time(predict(model_tuned, newdata = testData, type = "prob"))
size_base <- format(object.size(model_caro),  units = "Mb")
size_tune <- format(object.size(model_tuned), units = "Mb")

cat("\n=== Rendimiento computacional ===\n")
cat("Tiempo predicción base:    ", round(as.numeric(t_base["elapsed"]), 3), "s\n")
cat("Tiempo predicción tuneado: ", round(as.numeric(t_tune["elapsed"]), 3), "s\n")
cat("Tamaño modelo base:        ", size_base, "\n")
cat("Tamaño modelo tuneado:     ", size_tune, "\n")

## ========================================================
## ACTIVIDAD 9 - Tuneo del umbral con índice de Youden
## ========================================================
roc_obj <- roc(testData$caro_factor, pred_probs,
               levels = c("No", "Si"), direction = "<")
cat("\nAUC:", round(auc(roc_obj), 4), "\n")

best <- coords(roc_obj, x = "best", best.method = "youden",
               ret = c("threshold", "sensitivity", "specificity", "youden"))
opt_thr <- as.numeric(best$threshold[1])
cat("Umbral óptimo (Youden):", round(opt_thr, 4), "\n")

thresholds <- sort(unique(c(0.30, 0.35, 0.40, 0.45, 0.50, round(opt_thr, 3), 0.55, 0.60)))
tabla_thr  <- data.frame()
for (thr in thresholds) {
  pc <- factor(ifelse(pred_probs > thr, "Si", "No"), levels = c("No", "Si"))
  cm <- confusionMatrix(pc, testData$caro_factor, positive = "Si")
  tabla_thr <- rbind(tabla_thr, data.frame(
    Umbral = round(thr, 4),
    Accuracy = round(cm$overall["Accuracy"], 4),
    Sensibilidad = round(cm$byClass["Sensitivity"], 4),
    Especificidad = round(cm$byClass["Specificity"], 4),
    F1 = round(cm$byClass["F1"], 4),
    BalancedAcc = round(cm$byClass["Balanced Accuracy"], 4)
  ))
}
cat("\n=== Barrido de umbrales ===\n"); print(tabla_thr, row.names = FALSE)

pred_class_opt <- factor(ifelse(pred_probs > opt_thr, "Si", "No"), levels = c("No", "Si"))
conf_opt <- confusionMatrix(pred_class_opt, testData$caro_factor, positive = "Si")
print(conf_opt)

plot(roc_obj, main = "Curva ROC - Modelo Caro vs No Caro",
     col = "steelblue", lwd = 2, print.auc = TRUE)
points(as.numeric(best$specificity[1]), as.numeric(best$sensitivity[1]),
       pch = 19, col = "red", cex = 1.8)

## ========================================================
## ACTIVIDAD 10 - Selección del mejor modelo binario
## ========================================================
aic_base <- AIC(final_glm)
bic_base <- BIC(final_glm)

aic_glmnet <- function(model_train, x, y) {
  best_lambda <- model_train$bestTune$lambda
  coefs <- coef(model_train$finalModel, s = best_lambda)
  df    <- sum(as.numeric(coefs) != 0)
  linp  <- predict(model_train$finalModel, newx = x, s = best_lambda, type = "link")
  p     <- 1 / (1 + exp(-linp))
  p     <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  ll    <- sum(y * log(p) + (1 - y) * log(1 - p))
  n     <- length(y)
  list(AIC = -2 * ll + 2 * df, BIC = -2 * ll + log(n) * df, df = df)
}

xmat <- model.matrix(caro_factor ~ . - 1,
                     data = trainData[, c("caro_factor", predictors)])
yvec <- as.numeric(trainData$caro_factor == "Si")
aic_t <- aic_glmnet(model_tuned, xmat, yvec)

cat("\n=== AIC / BIC ===\n")
cat("GLM base     - AIC:", round(aic_base, 1), " BIC:", round(bic_base, 1), "\n")
cat("GLM tuneado  - AIC:", round(aic_t$AIC, 1), " BIC:", round(aic_t$BIC, 1),
    " (df:", aic_t$df, ")\n")

comparacion_bin <- data.frame(
  Modelo = c("GLM base (thr=0.5)", "GLM base (thr=Youden)", "GLM regularizado (thr=0.5)"),
  Accuracy      = c(conf_matrix$overall["Accuracy"], conf_opt$overall["Accuracy"], conf_tuned$overall["Accuracy"]),
  Sensibilidad  = c(conf_matrix$byClass["Sensitivity"], conf_opt$byClass["Sensitivity"], conf_tuned$byClass["Sensitivity"]),
  Especificidad = c(conf_matrix$byClass["Specificity"], conf_opt$byClass["Specificity"], conf_tuned$byClass["Specificity"]),
  Kappa         = c(conf_matrix$overall["Kappa"], conf_opt$overall["Kappa"], conf_tuned$overall["Kappa"]),
  F1            = c(conf_matrix$byClass["F1"], conf_opt$byClass["F1"], conf_tuned$byClass["F1"])
)
cat("\n=== Comparación final modelos binarios ===\n")
print(comparacion_bin, row.names = FALSE, digits = 4)

## ========================================================
## ACTIVIDAD 11 - Regresión Logística Multinomial
## ========================================================
## Clasificación directa en 3 clases: económico / medio / caro.
## Se tunea el parámetro 'decay' (regularización L2 de nnet).
## ========================================================

trainData$precio_cat <- factor(trainData$precio_cat,
                               levels = c("económico", "medio", "caro"))
testData$precio_cat  <- factor(testData$precio_cat,
                               levels = c("económico", "medio", "caro"))

# NB: en multiclase NO se puede usar twoClassSummary. Se usa defaultSummary.
trainControl_multi <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  savePredictions = TRUE
)

tuneGrid_multi <- expand.grid(decay = c(0, 0.001, 0.01, 0.1, 0.5, 1))

set.seed(123)
t_multi_train <- system.time({
  model_multi <- train(
    precio_cat ~ .,
    data      = trainData[, c("precio_cat", predictors)],
    method    = "multinom",
    trControl = trainControl_multi,
    tuneGrid  = tuneGrid_multi,
    trace     = FALSE,
    MaxNWts   = 6000,   # suficiente para las ~60 variables x 2 ecuaciones logit
    maxit     = 300
  )
})

cat("\n=== Modelo Multinomial - resumen del tuneo ===\n")
print(model_multi)
cat("\nMejor decay:", model_multi$bestTune$decay, "\n")
cat("Accuracy CV (mejor decay):",
    round(max(model_multi$results$Accuracy), 4), "\n")
cat("Tiempo de entrenamiento:", round(t_multi_train["elapsed"], 2), "s\n")

# Evaluación en test
t_multi_pred <- system.time({
  pred_multi <- predict(model_multi, newdata = testData)
})
conf_multi <- confusionMatrix(pred_multi, testData$precio_cat)
cat("\n=== Matriz de confusión - Multinomial ===\n")
print(conf_multi)

# Métricas por clase
metricas_por_clase_multi <- conf_multi$byClass[, c("Sensitivity", "Specificity",
                                                   "Precision", "F1")]
cat("\n=== Métricas por clase (multinomial) ===\n")
print(round(metricas_por_clase_multi, 4))

## ========================================================
## ACTIVIDAD 12 - Comparación con los demás algoritmos
## ========================================================
## Mismo train/test, mismos predictores -> comparación válida.
## ========================================================

form <- as.formula(paste("precio_cat ~", paste(predictors, collapse = " + ")))
xTrain <- trainData[, c("precio_cat", predictors)]
xTest  <- testData[,  c("precio_cat", predictors)]

# --- Árbol de decisión ---
set.seed(123)
t_tree_train <- system.time({
  mod_tree <- rpart(form, data = xTrain, method = "class")
})
t_tree_pred <- system.time({
  pred_tree <- predict(mod_tree, xTest, type = "class")
})
cm_tree <- confusionMatrix(pred_tree, testData$precio_cat)

# --- Random Forest (ntree=100 para que no tarde demasiado) ---
set.seed(123)
t_rf_train <- system.time({
  mod_rf <- randomForest(form, data = xTrain, ntree = 100, importance = FALSE)
})
t_rf_pred <- system.time({
  pred_rf <- predict(mod_rf, xTest)
})
cm_rf <- confusionMatrix(pred_rf, testData$precio_cat)

# --- Naive Bayes ---
set.seed(123)
t_nb_train <- system.time({
  mod_nb <- naiveBayes(form, data = xTrain)
})
t_nb_pred <- system.time({
  pred_nb <- predict(mod_nb, xTest)
})
cm_nb <- confusionMatrix(pred_nb, testData$precio_cat)

# --- KNN (con sólo variables numéricas escaladas) ---
num_pred <- c("accommodates", "bedrooms", "bathrooms",
              "review_scores_rating", "number_of_reviews")
xTr_num <- scale(trainData[, num_pred])
cent <- attr(xTr_num, "scaled:center")
scl  <- attr(xTr_num, "scaled:scale")
xTe_num <- scale(testData[, num_pred], center = cent, scale = scl)

set.seed(123)
t_knn_pred <- system.time({
  pred_knn <- class::knn(train = xTr_num, test = xTe_num,
                         cl = trainData$precio_cat, k = 11)
})
# KNN no tiene fase de entrenamiento separada; guardamos 0
t_knn_train <- c(elapsed = 0)
cm_knn <- confusionMatrix(pred_knn, testData$precio_cat)

# --- Multinomial ya entrenado arriba ---
cm_lr <- conf_multi

# --- Tabla final de comparación -----
tabla_final <- data.frame(
  Modelo         = c("Árbol Decisión", "Random Forest", "Naive Bayes",
                     "KNN (k=11)", "Reg. Log. Multinom."),
  Tiempo_train_s = round(c(t_tree_train["elapsed"], t_rf_train["elapsed"],
                           t_nb_train["elapsed"],  t_knn_train["elapsed"],
                           t_multi_train["elapsed"]), 2),
  Tiempo_pred_s  = round(c(t_tree_pred["elapsed"], t_rf_pred["elapsed"],
                           t_nb_pred["elapsed"],   t_knn_pred["elapsed"],
                           t_multi_pred["elapsed"]), 3),
  Accuracy       = round(c(cm_tree$overall["Accuracy"], cm_rf$overall["Accuracy"],
                           cm_nb$overall["Accuracy"],   cm_knn$overall["Accuracy"],
                           cm_lr$overall["Accuracy"]), 4),
  Kappa          = round(c(cm_tree$overall["Kappa"], cm_rf$overall["Kappa"],
                           cm_nb$overall["Kappa"],   cm_knn$overall["Kappa"],
                           cm_lr$overall["Kappa"]), 4)
)

# Agregamos el tamaño de cada modelo en memoria
tabla_final$Tamaño_MB <- c(
  round(as.numeric(object.size(mod_tree))   / 1024^2, 2),
  round(as.numeric(object.size(mod_rf))     / 1024^2, 2),
  round(as.numeric(object.size(mod_nb))     / 1024^2, 2),
  0,   # KNN no guarda un "modelo" per se
  round(as.numeric(object.size(model_multi))/ 1024^2, 2)
)

cat("\n=== Tabla comparativa final - TODOS los algoritmos (multiclase) ===\n")
print(tabla_final, row.names = FALSE)

# Guardar todo para el informe
save(model_caro, model_tuned, model_multi,
     mod_tree, mod_rf, mod_nb,
     conf_matrix, conf_opt, conf_tuned, conf_multi,
     cm_tree, cm_rf, cm_nb, cm_knn, cm_lr,
     tabla_final, comparacion_bin,
     trainData, testData, predictors,
     file = "resultados_finales.RData")

cat("\n=== Fin del script - Laboratorio 7 completo (Actividades 1 a 12) ===\n")
