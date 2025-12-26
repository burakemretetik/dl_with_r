# 1. SETUP ---------------------------------------------------------------------
source("cnn_lib.R")

set.seed(42)

TOTAL_IMAGES <- 500
TRAIN_SPLIT  <- 0.8
BATCH_SIZE   <- 32
EPOCHS       <- 10
LR           <- 0.001

# Loading data
data <- get_fashion_data()
X_total <- data$x_train[1:TOTAL_IMAGES, ]
Y_total <- data$y_train[, 1:TOTAL_IMAGES]
X_total <- array(t(X_total), dim = c(28, 28, 1, TOTAL_IMAGES))
X_total <- aperm(X_total, c(4, 1, 2, 3))

num_train <- floor(TOTAL_IMAGES * TRAIN_SPLIT)
X_train <- X_total[1:num_train, , , , drop=FALSE]
Y_train <- Y_total[, 1:num_train]
X_val   <- X_total[(num_train+1):TOTAL_IMAGES, , , , drop=FALSE]
Y_val   <- Y_total[, (num_train+1):TOTAL_IMAGES]

cat("Data Split:\nTrain:", dim(X_train)[1], "\nVal:  ", dim(X_val)[1], "\n")

# 2. INITIALIZATION ------------------------------------------------------------
# Initialize Parameters
params <- list(
  W_conv = array(runif(3*3*1*8, -0.2, 0.2), dim=c(3, 3, 1, 8)),
  W_fc   = matrix(runif(10 * (14 * 14 * 8), -0.05, 0.05), nrow=10, ncol=(14 * 14 * 8)),
  b_fc   = matrix(0, nrow=10, ncol=1)
)

# Initialize Adam Cache
adam_cache <- init_adam(params)
t <- 0 # Global time step

history <- data.frame(epoch=integer(), train_loss=numeric(), val_loss=numeric(),
                      train_acc=numeric(), val_acc=numeric())

# 3. TRAINING LOOP -------------------------------------------------------------
num_batches <- floor(num_train / BATCH_SIZE)
cat("\n=== Starting Adam Training ===\n")

for (e in 1:EPOCHS) {
  train_loss_sum <- 0
  train_acc_sum  <- 0

  indices <- sample(num_train)
  X_shuff <- X_train[indices, , , , drop=FALSE]
  Y_shuff <- Y_train[, indices]

  for (b in 1:num_batches) {
    t <- t + 1 # Increment time step for Adam

    # Slice
    start <- (b-1)*BATCH_SIZE + 1; end <- b*BATCH_SIZE
    bx <- X_shuff[start:end, , , , drop=FALSE]
    by <- Y_shuff[, start:end]

    # Forward
    conv_out <- conv2d_forward(bx, params$W_conv, stride=1, padding=1)
    relu_out <- ifelse(conv_out < 0, 0, conv_out)
    pool_out <- max_pool2d(relu_out, kernel_size=2, stride=2)

    flat_res <- flatten_forward(pool_out)
    fc_input <- flat_res$out
    Z_fc     <- params$W_fc %*% t(fc_input) + as.vector(params$b_fc)
    log_p    <- logsoftmax(Z_fc)

    # Metrics
    train_loss_sum <- train_loss_sum + cross_entropy_loss(by, log_p)
    preds <- max.col(t(Z_fc)) - 1
    truth <- max.col(t(by)) - 1
    train_acc_sum <- train_acc_sum + mean(preds == truth)

    # Backward
    dZ_fc <- exp(log_p) - by
    dW_fc <- (1/BATCH_SIZE) * (dZ_fc %*% fc_input)
    db_fc <- (1/BATCH_SIZE) * rowSums(dZ_fc)

    d_flat <- t(params$W_fc) %*% dZ_fc
    d_pool <- flatten_backward(t(d_flat), flat_res$original_dims)
    d_relu <- max_pool_backward(d_pool, relu_out, kernel_size=2, stride=2)
    d_conv <- d_relu; d_conv[conv_out < 0] <- 0

    conv_grads <- conv2d_backward(d_conv, bx, params$W_conv, stride=1, padding=1)
    dW_conv <- conv_grads$dW

    # --- ADAM UPDATE ---
    # Update W_conv
    upd_conv <- update_adam(params$W_conv, dW_conv, adam_cache$m$W_conv, adam_cache$v$W_conv, t, lr=LR)
    params$W_conv <- upd_conv$param; adam_cache$m$W_conv <- upd_conv$m; adam_cache$v$W_conv <- upd_conv$v

    # Update W_fc
    upd_fc <- update_adam(params$W_fc, dW_fc, adam_cache$m$W_fc, adam_cache$v$W_fc, t, lr=LR)
    params$W_fc <- upd_fc$param; adam_cache$m$W_fc <- upd_fc$m; adam_cache$v$W_fc <- upd_fc$v

    # Update b_fc
    upd_b <- update_adam(params$b_fc, as.matrix(db_fc), adam_cache$m$b_fc, adam_cache$v$b_fc, t, lr=LR)
    params$b_fc <- upd_b$param; adam_cache$m$b_fc <- upd_b$m; adam_cache$v$b_fc <- upd_b$v
  }

  # --- VALIDATION (Forward Only) ---
  v_conv <- conv2d_forward(X_val, params$W_conv, stride=1, padding=1)
  v_relu <- ifelse(v_conv < 0, 0, v_conv)
  v_pool <- max_pool2d(v_relu, kernel_size=2, stride=2)
  v_flat <- flatten_forward(v_pool)$out
  v_Z    <- params$W_fc %*% t(v_flat) + as.vector(params$b_fc)
  v_log  <- logsoftmax(v_Z)

  val_loss <- cross_entropy_loss(Y_val, v_log)
  v_preds  <- max.col(t(v_Z)) - 1
  v_truth  <- max.col(t(Y_val)) - 1
  val_acc  <- mean(v_preds == v_truth)

  # Logging
  avg_train_loss <- train_loss_sum / num_batches
  avg_train_acc  <- train_acc_sum / num_batches
  history[e, ] <- c(e, avg_train_loss, val_loss, avg_train_acc, val_acc)

  cat(sprintf("Epoch %02d | Train Loss: %.3f (Acc: %.0f%%) | Val Loss: %.3f (Acc: %.0f%%)\n",
              e, avg_train_loss, avg_train_acc*100, val_loss, val_acc*100))
}

# 4. VISUALIZATION -------------------------------------------------------------
par(mfrow=c(1,2))
# Plot Loss
y_range <- range(c(history$train_loss, history$val_loss))
plot(history$epoch, history$train_loss, type="o", col="blue", lwd=2, ylim=y_range,
     xlab="Epoch", ylab="Loss", main="Adam Loss Curve")
lines(history$epoch, history$val_loss, type="o", col="red", lwd=2)
legend("topright", legend=c("Train", "Val"), col=c("blue", "red"), lwd=2)

# Plot Accuracy
plot(history$epoch, history$train_acc, type="o", col="blue", lwd=2, ylim=c(0,1),
     xlab="Epoch", ylab="Accuracy", main="Adam Accuracy Curve")
lines(history$epoch, history$val_acc, type="o", col="red", lwd=2)
legend("bottomright", legend=c("Train", "Val"), col=c("blue", "red"), lwd=2)
