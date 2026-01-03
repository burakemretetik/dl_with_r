# LeNet with batch norm layers (Optimizer: SGD with Momentum)
# 1. SETUP ---------------------------------------------------------------------
source("cnn_lib.R")

set.seed(42)

TOTAL_IMAGES <- 500
TRAIN_SPLIT  <- 0.8
BATCH_SIZE   <- 32
EPOCHS       <- 10
LR           <- 0.01
MOMENTUM     <- 0.9

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

# Architecture:
# Conv1 (5x5, 6ch) -> BN -> ReLU -> AvgPool
# Conv2 (5x5, 16ch) -> BN -> ReLU -> AvgPool
# MLP (400 -> 120 -> 84 -> 10)

params <- list(
  # Layer 1
  W1 = array(runif(5*5*1*6, -0.1, 0.1), dim=c(5, 5, 1, 6)),
  gamma1 = rep(1, 6),
  beta1  = rep(0, 6),

  # Layer 2
  W2 = array(runif(5*5*6*16, -0.1, 0.1), dim=c(5, 5, 6, 16)),
  gamma2 = rep(1, 16),
  beta2  = rep(0, 16),

  # MLP
  # 5x5x16 = 400 features
  W3 = matrix(runif(120 * 400, -0.05, 0.05), nrow=120, ncol=400),
  b3 = matrix(0, nrow=120, ncol=1),

  W4 = matrix(runif(84 * 120, -0.05, 0.05), nrow=84, ncol=120),
  b4 = matrix(0, nrow=84, ncol=1),

  W5 = matrix(runif(10 * 84, -0.05, 0.05), nrow=10, ncol=84),
  b5 = matrix(0, nrow=10, ncol=1)
)

# Batch Norm Running Stats
bn_params <- list(
  bn1 = list(mode="train", running_mean=rep(0, 6), running_var=rep(1, 6), momentum=0.9),
  bn2 = list(mode="train", running_mean=rep(0, 16), running_var=rep(1, 16), momentum=0.9)
)

# Initialize SGD Momentum Cache
sgd_v <- init_sgd_momentum(params)

history <- data.frame(epoch=integer(), train_loss=numeric(), val_loss=numeric(),
                      train_acc=numeric(), val_acc=numeric())

# 3. TRAINING LOOP -------------------------------------------------------------
num_batches <- floor(num_train / BATCH_SIZE)
cat("\n=== Starting Training (LeNet + BN + SGD) ===\n")

for (e in 1:EPOCHS) {
  train_loss_sum <- 0
  train_acc_sum  <- 0

  indices <- sample(num_train)
  X_shuff <- X_train[indices, , , , drop=FALSE]
  Y_shuff <- Y_train[, indices]

  # Set Train Mode
  bn_params$bn1$mode <- "train"
  bn_params$bn2$mode <- "train"

  for (b in 1:num_batches) {
    # Slice
    start <- (b-1)*BATCH_SIZE + 1; end <- b*BATCH_SIZE
    bx <- X_shuff[start:end, , , , drop=FALSE]
    by <- Y_shuff[, start:end]

    # --- FORWARD PASS ---

    # L1: Conv -> BN -> ReLU -> AvgPool
    z1 <- conv2d_forward(bx, params$W1, stride=1, padding=2)
    bn1_out <- batch_norm_forward(z1, params$gamma1, params$beta1, bn_params$bn1)
    a1 <- relu(bn1_out$out)
    p1 <- avg_pool2d(a1, kernel_size=2, stride=2)

    bn_params$bn1 <- bn1_out$bn_param # Update running stats

    # L2: Conv -> BN -> ReLU -> AvgPool
    z2 <- conv2d_forward(p1, params$W2, stride=1, padding=0)
    bn2_out <- batch_norm_forward(z2, params$gamma2, params$beta2, bn_params$bn2)
    a2 <- relu(bn2_out$out)
    p2 <- avg_pool2d(a2, kernel_size=2, stride=2)

    bn_params$bn2 <- bn2_out$bn_param # Update running stats

    # MLP
    flat <- flatten_forward(p2)
    fc_in <- flat$out

    z3 <- params$W3 %*% t(fc_in) + as.vector(params$b3)
    a3 <- relu(z3)

    z4 <- params$W4 %*% a3 + as.vector(params$b4)
    a4 <- relu(z4)

    z5 <- params$W5 %*% a4 + as.vector(params$b5)
    log_p <- logsoftmax(z5)

    # Metrics
    train_loss_sum <- train_loss_sum + cross_entropy_loss(by, log_p)
    preds <- max.col(t(z5)) - 1
    truth <- max.col(t(by)) - 1
    train_acc_sum <- train_acc_sum + mean(preds == truth)

    # --- BACKWARD PASS ---

    # Output
    dZ5 <- exp(log_p) - by
    dW5 <- (1/BATCH_SIZE) * (dZ5 %*% t(a4))
    db5 <- (1/BATCH_SIZE) * rowSums(dZ5)

    # Hidden 2
    dA4 <- t(params$W5) %*% dZ5
    dZ4 <- relu_backward(dA4, z4)
    dW4 <- (1/BATCH_SIZE) * (dZ4 %*% t(a3))
    db4 <- (1/BATCH_SIZE) * rowSums(dZ4)

    # Hidden 1
    dA3 <- t(params$W4) %*% dZ4
    dZ3 <- relu_backward(dA3, z3)
    dW3 <- (1/BATCH_SIZE) * (dZ3 %*% fc_in)
    db3 <- (1/BATCH_SIZE) * rowSums(dZ3)

    # Flatten Back
    d_flat <- t(params$W3) %*% dZ3
    dP2 <- flatten_backward(t(d_flat), flat$original_dims)

    # L2 Backward
    dA2 <- avg_pool_backward(dP2, a2, kernel_size=2, stride=2)
    dBN2 <- relu_backward(dA2, bn2_out$out)

    bn2_grads <- batch_norm_backward(dBN2, bn2_out$cache)
    dGamma2 <- bn2_grads$dgamma
    dBeta2  <- bn2_grads$dbeta
    dZ2     <- bn2_grads$dX

    conv2_grads <- conv2d_backward(dZ2, p1, params$W2, stride=1, padding=0)
    dW2 <- conv2_grads$dW
    dP1 <- conv2_grads$dX

    # L1 Backward
    dA1 <- avg_pool_backward(dP1, a1, kernel_size=2, stride=2)
    dBN1 <- relu_backward(dA1, bn1_out$out)

    bn1_grads <- batch_norm_backward(dBN1, bn1_out$cache)
    dGamma1 <- bn1_grads$dgamma
    dBeta1  <- bn1_grads$dbeta
    dZ1     <- bn1_grads$dX

    conv1_grads <- conv2d_backward(dZ1, bx, params$W1, stride=1, padding=2)
    dW1 <- conv1_grads$dW

    # --- UPDATE (SGD Momentum) ---
    grads <- list(
      W1=dW1, gamma1=dGamma1, beta1=dBeta1,
      W2=dW2, gamma2=dGamma2, beta2=dBeta2,
      W3=dW3, b3=db3, W4=dW4, b4=db4, W5=dW5, b5=db5
    )

    for(p in names(params)) {
      res <- update_sgd_momentum(params[[p]], grads[[p]], sgd_v[[p]], lr=LR, momentum=MOMENTUM)
      params[[p]] <- res$param
      sgd_v[[p]]  <- res$v
    }
  }

  # --- VALIDATION (Forward Only) ---
  bn_params$bn1$mode <- "test"
  bn_params$bn2$mode <- "test"

  # L1
  z1_v <- conv2d_forward(X_val, params$W1, stride=1, padding=2)
  bn1_v <- batch_norm_forward(z1_v, params$gamma1, params$beta1, bn_params$bn1)
  p1_v <- avg_pool2d(relu(bn1_v$out), 2, 2)

  # L2
  z2_v <- conv2d_forward(p1_v, params$W2, stride=1, padding=0)
  bn2_v <- batch_norm_forward(z2_v, params$gamma2, params$beta2, bn_params$bn2)
  p2_v <- avg_pool2d(relu(bn2_v$out), 2, 2)

  # MLP
  flat_v <- flatten_forward(p2_v)
  z3_v <- params$W3 %*% t(flat_v$out) + as.vector(params$b3)
  a3_v <- relu(z3_v)
  z4_v <- params$W4 %*% a3_v + as.vector(params$b4)
  a4_v <- relu(z4_v)
  z5_v <- params$W5 %*% a4_v + as.vector(params$b5)
  log_p_v <- logsoftmax(z5_v)

  val_loss <- cross_entropy_loss(Y_val, log_p_v)
  v_preds <- max.col(t(z5_v)) - 1
  v_truth <- max.col(t(Y_val)) - 1
  val_acc <- mean(v_preds == v_truth)

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
     xlab="Epoch", ylab="Loss", main="LeNet-BN Loss")
lines(history$epoch, history$val_loss, type="o", col="red", lwd=2)
legend("topright", legend=c("Train", "Val"), col=c("blue", "red"), lwd=2)

# Plot Accuracy
plot(history$epoch, history$train_acc, type="o", col="blue", lwd=2, ylim=c(0,1),
     xlab="Epoch", ylab="Accuracy", main="LeNet-BN Accuracy")
lines(history$epoch, history$val_acc, type="o", col="red", lwd=2)
legend("bottomright", legend=c("Train", "Val"), col=c("blue", "red"), lwd=2)

