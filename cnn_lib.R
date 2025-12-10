# GET FASHION DATA

get_fashion_data <- function() {
  # NOTE: Ensure these 4 files are in your getwd() folder!
  x_train <- read_idx_file("train-images-idx3-ubyte.gz")
  y_train_raw <- read_idx_file("train-labels-idx1-ubyte.gz")
  y_train <- one_hot_encode(y_train_raw)

  x_val <- read_idx_file("t10k-images-idx3-ubyte.gz")
  y_val_raw <- read_idx_file("t10k-labels-idx1-ubyte.gz")
  y_val <- one_hot_encode(y_val_raw)

  return(list(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val))
}

# ==============================================================================
# PART 1: ACTIVATIONS & LOSS
# ==============================================================================

relu <- function(x) { ifelse(x < 0, 0, x) }

logsoftmax <- function(x) {
  # Numerically stable softmax
  x_max <- apply(x, 2, max)
  x_shifted <- sweep(x, 2, x_max, "-")
  lse <- log(colSums(exp(x_shifted)))
  return(sweep(x_shifted, 2, lse, "-"))
}

cross_entropy_loss <- function(Y, logprobs) {
  # Y: One-hot [Classes, Batch]
  # logprobs: [Classes, Batch]
  m <- ncol(logprobs)
  loss <- -sum(Y * logprobs) / m
  return(loss)
}

# ==============================================================================
# PART 2: CONVOLUTION LAYER
# ==============================================================================

conv2d_forward <- function(X, W, stride=1, padding=1) {
  # Input Dimensions
  N <- dim(X)[1]; H_in <- dim(X)[2]; W_in <- dim(X)[3]; C_in <- dim(X)[4]
  K_h <- dim(W)[1]; K_w <- dim(W)[2]; C_out <- dim(W)[4]

  # Output Dimensions
  H_out <- floor((H_in - K_h + 2*padding) / stride) + 1
  W_out <- floor((W_in - K_w + 2*padding) / stride) + 1
  Z <- array(0, dim = c(N, H_out, W_out, C_out))

  # Padding
  if (padding > 0) {
    X_pad <- array(0, dim = c(N, H_in + 2*padding, W_in + 2*padding, C_in))
    X_pad[, (padding+1):(padding+H_in), (padding+1):(padding+W_in), ] <- X
  } else { X_pad <- X }

  # Convolution Loop
  for (n in 1:N) {
    for (d in 1:C_out) {
      for (i in 1:H_out) {
        for (j in 1:W_out) {
          h_start <- (i-1)*stride + 1; h_end <- h_start + K_h - 1
          w_start <- (j-1)*stride + 1; w_end <- w_start + K_w - 1

          patch <- X_pad[n, h_start:h_end, w_start:w_end, ]
          filter <- W[, , , d]
          Z[n, i, j, d] <- sum(patch * filter)
        }
      }
    }
  }
  return(Z)
}

conv2d_backward <- function(dZ, X, W, stride=1, padding=1) {
  # Setup Dimensions
  N <- dim(X)[1]; H_in <- dim(X)[2]; W_in <- dim(X)[3]; C_in <- dim(X)[4]
  K_h <- dim(W)[1]; K_w <- dim(W)[2]; C_out <- dim(W)[4]
  H_out <- dim(dZ)[2]; W_out <- dim(dZ)[3]

  # Init Gradients
  dW <- array(0, dim = dim(W))
  dX_padded <- array(0, dim = c(N, H_in + 2*padding, W_in + 2*padding, C_in))

  # Pad Input for lookup
  if (padding > 0) {
    X_pad <- array(0, dim = c(N, H_in + 2*padding, W_in + 2*padding, C_in))
    X_pad[, (padding+1):(padding+H_in), (padding+1):(padding+W_in), ] <- X
  } else { X_pad <- X }

  # Gradient Accumulation Loop
  for (n in 1:N) {
    for (d in 1:C_out) {
      for (i in 1:H_out) {
        for (j in 1:W_out) {
          current_grad <- dZ[n, i, j, d]
          if (current_grad == 0) next

          h_start <- (i-1)*stride + 1; h_end <- h_start + K_h - 1
          w_start <- (j-1)*stride + 1; w_end <- w_start + K_w - 1

          # dW += Input * Grad
          patch <- X_pad[n, h_start:h_end, w_start:w_end, ]
          dW[, , , d] <- dW[, , , d] + (patch * current_grad)

          # dX += Weight * Grad
          filter <- W[, , , d]
          dX_padded[n, h_start:h_end, w_start:w_end, ] <-
            dX_padded[n, h_start:h_end, w_start:w_end, ] + (filter * current_grad)
        }
      }
    }
  }

  # Remove Padding
  if (padding > 0) {
    dX <- dX_padded[, (padding+1):(padding+H_in), (padding+1):(padding+W_in), ]
  } else { dX <- dX_padded }

  return(list(dX=dX, dW=dW))
}

# ==============================================================================
# PART 3: POOLING LAYER
# ==============================================================================

max_pool2d <- function(X, kernel_size=2, stride=2) {
  N <- dim(X)[1]; H_in <- dim(X)[2]; W_in <- dim(X)[3]; C <- dim(X)[4]
  H_out <- floor((H_in - kernel_size) / stride) + 1
  W_out <- floor((W_in - kernel_size) / stride) + 1

  Z <- array(0, dim = c(N, H_out, W_out, C))

  for (n in 1:N) {
    for (c in 1:C) {
      for (i in 1:H_out) {
        for (j in 1:W_out) {
          h_start <- (i-1)*stride + 1; h_end <- h_start + kernel_size - 1
          w_start <- (j-1)*stride + 1; w_end <- w_start + kernel_size - 1

          patch <- X[n, h_start:h_end, w_start:w_end, c]
          Z[n, i, j, c] <- max(patch)
        }
      }
    }
  }
  return(Z)
}

max_pool_backward <- function(dZ, X_orig, kernel_size=2, stride=2) {
  dX <- array(0, dim = dim(X_orig))
  N <- dim(dX)[1]; H_out <- dim(dZ)[2]; W_out <- dim(dZ)[3]; C <- dim(dX)[4]

  for (n in 1:N) {
    for (c in 1:C) {
      for (i in 1:H_out) {
        for (j in 1:W_out) {
          h_start <- (i-1)*stride + 1; h_end <- h_start + kernel_size - 1
          w_start <- (j-1)*stride + 1; w_end <- w_start + kernel_size - 1

          patch <- X_orig[n, h_start:h_end, w_start:w_end, c]
          max_idx <- which.max(patch)

          # Create mask for the winner
          mask <- matrix(0, nrow=kernel_size, ncol=kernel_size)
          mask[max_idx] <- 1

          # Accumulate gradient
          current_grad <- dZ[n, i, j, c]
          dX[n, h_start:h_end, w_start:w_end, c] <-
            dX[n, h_start:h_end, w_start:w_end, c] + (mask * current_grad)
        }
      }
    }
  }
  return(dX)
}

# ==============================================================================
# PART 4: FLATTEN UTILS
# ==============================================================================

flatten_forward <- function(X) {
  dims <- dim(X)
  N <- dims[1]
  # Flatten to [N, Features]
  out <- matrix(as.vector(X), nrow=N, byrow=FALSE)
  return(list(out=out, original_dims=dims))
}

flatten_backward <- function(dZ, original_dims) {
  return(array(dZ, dim=original_dims))
}
