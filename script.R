## ========================================================
## LABORATORIO 7 - Regresión Logística
## SmartStay Advisors - Optimización de precios en Airbnb
## COMMIT: Actividades 1 a 10 (modelo binario completo)
## ========================================================

library(caret)
library(car)
library(ggplot2)
library(pROC)
library(glmnet)
library(profvis)
library(htmlwidgets)

# ====================== ACTIVIDAD 1 ======================
# --- Carga y limpieza de datos --------------------------
load("listings.RData")

# Limpiar columna price (quitar $ y comas)
listings$price <- as.numeric(gsub("[$,]", "", listings$price))

# Recrear precio_cat usando cuartiles (igual que en laboratorios anteriores)
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

# Tres variables dicotómicas solicitadas
listings$is_economico <- ifelse(listings$precio_cat == "económico", 1, 0)
listings$is_medio     <- ifelse(listings$precio_cat == "medio",     1, 0)
listings$is_caro      <- ifelse(listings$precio_cat == "caro",      1, 0)

cat("\nDistribución de precio_cat:\n")
print(table(listings$precio_cat))

# ====================== ACTIVIDAD 2 ======================
# --- Misma partición 70/30 que en laboratorios anteriores
set.seed(123)
index <- createDataPartition(listings$precio_cat, p = 0.7, list = FALSE)
trainData <- listings[index, ]
testData  <- listings[-index, ]

# Predictores (mismos que en KNN, Árbol, RF y NB)
predictors <- c("room_type", "accommodates", "bedrooms", "bathrooms",
                "review_scores_rating", "number_of_reviews",
                "host_is_superhost", "instant_bookable",
                "neighbourhood_cleansed", "property_type")

vars_modelo <- c("is_caro", predictors)

# Eliminar NA en variables usadas por el modelo
trainData <- trainData[complete.cases(trainData[, vars_modelo]), ]
testData  <- testData[complete.cases(testData[,  vars_modelo]), ]

# Filtrar niveles no vistos en test (evita error en predict)
testData <- testData[
  testData$neighbourhood_cleansed %in% levels(factor(trainData$neighbourhood_cleansed)) &
    testData$property_type %in% levels(factor(trainData$property_type)) &
    testData$room_type %in% levels(factor(trainData$room_type)), ]

cat("\nFilas finales - Train:", nrow(trainData), " | Test:", nrow(testData), "\n")

# ====================== ACTIVIDAD 3 ======================
# --- Modelo base: regresión logística con 10-fold CV -----
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
summary(model_caro$finalModel)

# ====================== ACTIVIDAD 4 ======================
# --- Diagnóstico del modelo -----------------------------
final_glm <- model_caro$finalModel

cat("\n=== Variables aliadas (multicolinealidad perfecta) ===\n")
aliased <- attributes(alias(final_glm))$Complete
if (!is.null(aliased)) print(rownames(aliased)) else cat("No se detectaron variables aliadas\n")

cat("\n=== Coeficientes significativos (p < 0.05) ===\n")
coef_table <- summary(final_glm)$coefficients
print(head(coef_table[coef_table[, 4] < 0.05, ], 15))

cat("\n=== Matriz de correlación de predictores numéricos ===\n")
num_vars <- c("accommodates", "bedrooms", "bathrooms",
              "review_scores_rating", "number_of_reviews")
cor_matrix <- cor(trainData[, num_vars], use = "complete.obs")
print(round(cor_matrix, 2))

pseudo_r2 <- 1 - (final_glm$deviance / final_glm$null.deviance)
cat("\nPseudo R² (McFadden) =", round(pseudo_r2, 4), "\n")

# ====================== ACTIVIDAD 5 ======================
# --- Predicciones sobre el conjunto de prueba -----------
pred_probs <- predict(model_caro, newdata = testData, type = "prob")[, "Si"]
pred_class <- factor(ifelse(pred_probs > 0.5, "Si", "No"), levels = c("No", "Si"))

conf_matrix <- confusionMatrix(pred_class, testData$caro_factor, positive = "Si")
print(conf_matrix)

# ====================== ACTIVIDAD 6 ======================
# --- Análisis de sobreajuste ----------------------------
train_probs <- predict(model_caro, newdata = trainData, type = "prob")[, "Si"]
train_class <- factor(ifelse(train_probs > 0.5, "Si", "No"), levels = c("No", "Si"))

train_error <- 1 - mean(train_class == trainData$caro_factor)
test_error  <- 1 - conf_matrix$overall["Accuracy"]
cat("\nError entrenamiento:", round(train_error, 4),
    " | Error prueba:", round(test_error, 4), "\n")

# --- Curva de aprendizaje (real: un GLM por tamaño) -----
set.seed(123)
train_sizes   <- round(seq(0.2, 1, by = 0.1) * nrow(trainData))
train_errors  <- numeric(length(train_sizes))
test_errors   <- numeric(length(train_sizes))

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
       y = "Error de clasificación",
       color = "") +
  theme_minimal() +
  scale_color_manual(values = c("Entrenamiento" = "blue", "Prueba" = "red"))
print(p_learn)

## ========================================================
## ACTIVIDAD 7 - Tuneo del modelo (Regularización)
## ========================================================
## Se explora Ridge (alpha=0), Lasso (alpha=1) y Elastic Net
## (alpha intermedio) con varios valores de lambda.
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

cat("\n=== Mejor combinación encontrada ===\n")
print(model_tuned$bestTune)
cat("Mejor ROC (tuneado): ", round(max(model_tuned$results$ROC), 4), "\n")
cat("Mejor ROC (base):    ", round(max(model_caro$results$ROC),  4), "\n")

# Evaluación del modelo tuneado en test
pred_probs_tuned <- predict(model_tuned, newdata = testData, type = "prob")[, "Si"]
pred_class_tuned <- factor(ifelse(pred_probs_tuned > 0.5, "Si", "No"),
                           levels = c("No", "Si"))
conf_tuned <- confusionMatrix(pred_class_tuned, testData$caro_factor, positive = "Si")
cat("\n=== Matriz de confusión - Modelo tuneado (thr=0.5) ===\n")
print(conf_tuned)

# Visualización del comportamiento del hiperparámetro
p_tune <- ggplot(model_tuned$results,
                 aes(x = lambda, y = ROC, color = factor(alpha))) +
  geom_line() + geom_point(size = 1) +
  scale_x_log10() +
  labs(title = "Tuneo de hiperparámetros - ROC por alpha y lambda",
       x = "lambda (escala log)", y = "ROC (10-fold CV)",
       color = "alpha") +
  theme_minimal()
print(p_tune)

## ========================================================
## ACTIVIDAD 8 - Matriz de confusión + profiler
## ========================================================

# --- Profiling del pipeline de predicción con profvis ----
prof_base <- profvis({
  pp_b <- predict(model_caro, newdata = testData, type = "prob")[, "Si"]
  pc_b <- factor(ifelse(pp_b > 0.5, "Si", "No"), levels = c("No", "Si"))
  cm_b <- confusionMatrix(pc_b, testData$caro_factor, positive = "Si")
})
# Guardar el reporte HTML del profiler
htmlwidgets::saveWidget(prof_base, "profiler_base.html", selfcontained = TRUE)

prof_tuned <- profvis({
  pp_t <- predict(model_tuned, newdata = testData, type = "prob")[, "Si"]
  pc_t <- factor(ifelse(pp_t > 0.5, "Si", "No"), levels = c("No", "Si"))
  cm_t <- confusionMatrix(pc_t, testData$caro_factor, positive = "Si")
})
htmlwidgets::saveWidget(prof_tuned, "profiler_tuned.html", selfcontained = TRUE)

# Tiempos directos y tamaño en memoria
t_base <- system.time(predict(model_caro,  newdata = testData, type = "prob"))
t_tune <- system.time(predict(model_tuned, newdata = testData, type = "prob"))

size_base <- format(object.size(model_caro),  units = "Mb")
size_tune <- format(object.size(model_tuned), units = "Mb")

cat("\n=== Rendimiento computacional ===\n")
cat("Tiempo predicción base:    ", round(as.numeric(t_base["elapsed"]), 3), "s\n")
cat("Tiempo predicción tuneado: ", round(as.numeric(t_tune["elapsed"]), 3), "s\n")
cat("Tamaño modelo base:        ", size_base, "\n")
cat("Tamaño modelo tuneado:     ", size_tune, "\n")

# Análisis de dónde se equivoca más
cm_tbl <- conf_matrix$table
cat("\n=== Distribución de errores (modelo base, thr=0.5) ===\n")
cat("  FP (predicho Si, real No):", cm_tbl["Si", "No"],
    "- cobrar sobreprecio injustificado\n")
cat("  FN (predicho No, real Si):", cm_tbl["No", "Si"],
    "- NO detectar propiedades caras (pierde oportunidad de incentivo Airbnb)\n")

## ========================================================
## ACTIVIDAD 9 - Tuneo del umbral con índice de Youden
## ========================================================

roc_obj <- roc(testData$caro_factor, pred_probs,
               levels = c("No", "Si"), direction = "<")
cat("\nAUC:", round(auc(roc_obj), 4), "\n")

# Punto óptimo por índice de Youden
best <- coords(roc_obj, x = "best", best.method = "youden",
               ret = c("threshold", "sensitivity", "specificity", "youden"))
print(best)
opt_thr <- as.numeric(best$threshold[1])
cat("Umbral óptimo (Youden):", round(opt_thr, 4), "\n")

# Barrido exploratorio de umbrales
thresholds <- sort(unique(c(0.30, 0.35, 0.40, 0.45, 0.50, round(opt_thr, 3), 0.55, 0.60)))
tabla_thr  <- data.frame()
for (thr in thresholds) {
  pc <- factor(ifelse(pred_probs > thr, "Si", "No"), levels = c("No", "Si"))
  cm <- confusionMatrix(pc, testData$caro_factor, positive = "Si")
  tabla_thr <- rbind(tabla_thr, data.frame(
    Umbral        = round(thr, 4),
    Accuracy      = round(cm$overall["Accuracy"], 4),
    Sensibilidad  = round(cm$byClass["Sensitivity"], 4),
    Especificidad = round(cm$byClass["Specificity"], 4),
    F1            = round(cm$byClass["F1"], 4),
    BalancedAcc   = round(cm$byClass["Balanced Accuracy"], 4)
  ))
}
cat("\n=== Comparación de umbrales ===\n")
print(tabla_thr, row.names = FALSE)

# Matriz de confusión con umbral óptimo
pred_class_opt <- factor(ifelse(pred_probs > opt_thr, "Si", "No"), levels = c("No", "Si"))
conf_opt <- confusionMatrix(pred_class_opt, testData$caro_factor, positive = "Si")
cat("\n=== Matriz de confusión - Umbral Youden ===\n")
print(conf_opt)

# Gráfica ROC con punto óptimo
plot(roc_obj, main = "Curva ROC - Modelo Caro vs No Caro",
     col = "steelblue", lwd = 2, print.auc = TRUE)
points(as.numeric(best$specificity[1]), as.numeric(best$sensitivity[1]),
       pch = 19, col = "red", cex = 1.8)

## ========================================================
## ACTIVIDAD 10 - Selección del mejor modelo binario
## ========================================================

# AIC / BIC del modelo base (glm puro)
aic_base <- AIC(final_glm)
bic_base <- BIC(final_glm)

# AIC/BIC aproximado para glmnet (usando grados de libertad efectivos)
aic_glmnet <- function(model_train, x, y) {
  best_lambda <- model_train$bestTune$lambda
  coefs <- coef(model_train$finalModel, s = best_lambda)
  df    <- sum(as.numeric(coefs) != 0)
  linp  <- predict(model_train$finalModel, newx = x, s = best_lambda, type = "link")
  p     <- 1 / (1 + exp(-linp))
  p     <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  ll    <- sum(y * log(p) + (1 - y) * log(1 - p))
  n     <- length(y)
  list(AIC = -2 * ll + 2 * df,
       BIC = -2 * ll + log(n) * df,
       df  = df)
}

# Construir matriz de diseño para glmnet (igual que usa caret)
xmat <- model.matrix(caro_factor ~ . - 1,
                     data = trainData[, c("caro_factor", predictors)])
yvec <- as.numeric(trainData$caro_factor == "Si")
aic_t <- aic_glmnet(model_tuned, xmat, yvec)

cat("\n=== Comparación AIC / BIC ===\n")
cat("GLM base     - AIC:", round(aic_base, 1), " BIC:", round(bic_base, 1), "\n")
cat("GLM tuneado  - AIC:", round(aic_t$AIC, 1), " BIC:", round(aic_t$BIC, 1),
    " (df efectivos:", aic_t$df, ")\n")

# Tabla resumen final de los 3 modelos binarios
comparacion_bin <- data.frame(
  Modelo = c("GLM base (thr=0.5)",
             "GLM base (thr=Youden)",
             "GLM regularizado (thr=0.5)"),
  Accuracy      = c(conf_matrix$overall["Accuracy"],
                    conf_opt$overall["Accuracy"],
                    conf_tuned$overall["Accuracy"]),
  Sensibilidad  = c(conf_matrix$byClass["Sensitivity"],
                    conf_opt$byClass["Sensitivity"],
                    conf_tuned$byClass["Sensitivity"]),
  Especificidad = c(conf_matrix$byClass["Specificity"],
                    conf_opt$byClass["Specificity"],
                    conf_tuned$byClass["Specificity"]),
  Kappa         = c(conf_matrix$overall["Kappa"],
                    conf_opt$overall["Kappa"],
                    conf_tuned$overall["Kappa"]),
  F1            = c(conf_matrix$byClass["F1"],
                    conf_opt$byClass["F1"],
                    conf_tuned$byClass["F1"])
)
cat("\n=== Tabla comparativa final - modelos binarios ===\n")
print(comparacion_bin, row.names = FALSE, digits = 4)

# Guardar objetos clave para próxima entrega (commit 2)
save(model_caro, model_tuned, conf_matrix, conf_opt, conf_tuned,
     trainData, testData, predictors, pred_probs, opt_thr,
     file = "modelos_binarios.RData")

cat("\n=== Fin del script (Actividades 1 a 10) ===\n")
