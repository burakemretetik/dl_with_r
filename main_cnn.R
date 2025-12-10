# 1. Setup
source("cnn_lib.R")

set.seed(42)

# --- Configuration ---
TOTAL_IMAGES <- 500
TRAIN_SPLIT  <- 0.8  # 80% Train, 20% Val
BATCH_SIZE   <- 32
EPOCHS       <- 10
LR           <- 0.05

# --- Load & Preprocess ---
data <- get_fashion_data()

# Use 200 total images
X_total <- data$x_train[1:TOTAL_IMAGES, ]
Y_total <- data$y_train[, 1:TOTAL_IMAGES]

# Reshape [N, 784] -> [N, 28, 28, 1]
X_total <- array(t(X_total), dim = c(28, 28, 1, TOTAL_IMAGES))
X_total <- aperm(X_total, c(4, 1, 2, 3))

# --- Split Train / Val ---
num_train <- floor(TOTAL_IMAGES * TRAIN_SPLIT)
num_val   <- TOTAL_IMAGES - num_train

# Train Set
X_train <- X_total[1:num_train, , , , drop=FALSE]
Y_train <- Y_total[, 1:num_train]

# Val Set
X_val   <- X_total[(num_train+1):TOTAL_IMAGES, , , , drop=FALSE]
Y_val   <- Y_total[, (num_train+1):TOTAL_IMAGES]

cat("Data Split:\n")
cat("Train:", dim(X_train)[1], "images\n")
cat("Val:  ", dim(X_val)[1], "images\n")

# --- Initialization ---
W_conv <- array(runif(3*3*1*8, -0.2, 0.2), dim=c(3, 3, 1, 8))
n_flat <- 14 * 14 * 8
W_fc <- matrix(runif(10 * n_flat, -0.05, 0.05), nrow=10, ncol=n_flat)
b_fc <- matrix(0, nrow=10, ncol=1)

# History Container
history <- data.frame(epoch=integer(),
                      train_loss=numeric(), val_loss=numeric(),
                      train_acc=numeric(), val_acc=numeric())

# --- Training Loop ---
num_batches <- floor(num_train / BATCH_SIZE)

cat("\n=== Starting Training ===\n")

for (e in 1:EPOCHS) {

  # --- 1. TRAINING PHASE ---
  train_loss_sum <- 0
  train_acc_sum  <- 0

  # Shuffle Train Data Only
  indices <- sample(num_train)
  X_shuff <- X_train[indices, , , , drop=FALSE]
  Y_shuff <- Y_train[, indices]

  for (b in 1:num_batches) {
    # Slice
    start <- (b-1)*BATCH_SIZE + 1; end <- b*BATCH_SIZE
    bx <- X_shuff[start:end, , , , drop=FALSE]
    by <- Y_shuff[, start:end]

    # Forward
    conv_out <- conv2d_forward(bx, W_conv, stride=1, padding=1)
    relu_out <- ifelse(conv_out < 0, 0, conv_out)
    pool_out <- max_pool2d(relu_out, kernel_size=2, stride=2)

    flat_res <- flatten_forward(pool_out)
    fc_input <- flat_res$out
    Z_fc     <- W_fc %*% t(fc_input) + as.vector(b_fc)
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

    d_flat <- t(W_fc) %*% dZ_fc
    d_pool <- flatten_backward(t(d_flat), flat_res$original_dims)
    d_relu <- max_pool_backward(d_pool, relu_out, kernel_size=2, stride=2)
    d_conv <- d_relu; d_conv[conv_out < 0] <- 0

    conv_grads <- conv2d_backward(d_conv, bx, W_conv, stride=1, padding=1)

    # Update
    W_fc   <- W_fc - LR * dW_fc
    b_fc   <- b_fc - LR * db_fc
    W_conv <- W_conv - LR * conv_grads$dW
  }

  # --- 2. VALIDATION PHASE (No Backprop) ---
  # Forward Pass on FULL Validation Set
  v_conv <- conv2d_forward(X_val, W_conv, stride=1, padding=1)
  v_relu <- ifelse(v_conv < 0, 0, v_conv)
  v_pool <- max_pool2d(v_relu, kernel_size=2, stride=2)
  v_flat <- flatten_forward(v_pool)$out
  v_Z    <- W_fc %*% t(v_flat) + as.vector(b_fc)
  v_log  <- logsoftmax(v_Z)

  # Val Metrics
  val_loss <- cross_entropy_loss(Y_val, v_log)
  v_preds  <- max.col(t(v_Z)) - 1
  v_truth  <- max.col(t(Y_val)) - 1
  val_acc  <- mean(v_preds == v_truth)

  # --- 3. LOGGING ---
  avg_train_loss <- train_loss_sum / num_batches
  avg_train_acc  <- train_acc_sum / num_batches

  history[e, ] <- c(e, avg_train_loss, val_loss, avg_train_acc, val_acc)

  cat(sprintf("Epoch %02d | Train Loss: %.3f (Acc: %.0f%%) | Val Loss: %.3f (Acc: %.0f%%)\n",
              e, avg_train_loss, avg_train_acc*100, val_loss, val_acc*100))
}

# --- 4. VISUALIZATION: The Learning Curves ---
# Plot Loss
par(mfrow=c(1,2)) # Two graphs side-by-side

# Graph A: Loss
y_min <- min(history$train_loss, history$val_loss)
y_max <- max(history$train_loss, history$val_loss)

plot(history$epoch, history$train_loss, type="o", col="blue", lwd=2,
     ylim=c(y_min, y_max), xlab="Epoch", ylab="Loss", main="Loss Curve")
lines(history$epoch, history$val_loss, type="o", col="red", lwd=2)
legend("topright", legend=c("Train", "Val"), col=c("blue", "red"), lwd=2)

# Graph B: Accuracy
plot(history$epoch, history$train_acc, type="o", col="blue", lwd=2,
     ylim=c(0, 1), xlab="Epoch", ylab="Accuracy", main="Accuracy Curve")
lines(history$epoch, history$val_acc, type="o", col="red", lwd=2)
legend("bottomright", legend=c("Train", "Val"), col=c("blue", "red"), lwd=2)

grid()
