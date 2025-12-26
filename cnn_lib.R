# ==============================================================================
# PART 0: DATA LOADING & PREPROCESSING
# ==============================================================================

get_fashion_data <- function() {
  # Adjust paths if necessary based on your folder structure
  x_train <- read_idx_file("../train-images-idx3-ubyte.gz")
  y_train_raw <- read_idx_file("../train-labels-idx1-ubyte.gz")
  y_train <- one_hot_encode(y_train_raw)

  x_val <- read_idx_file("../t10k-images-idx3-ubyte.gz")
  y_val_raw <- read_idx_file("../t10k-labels-idx1-ubyte.gz")
  y_val <- one_hot_encode(y_val_raw)
  return(list(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val))
}

read_idx_file <- function(filename) {
  # --- Robust Path Finding ---
  paths_to_check <- c(filename, file.path("data", filename), file.path("..", filename))
  found_path <- NULL
  for (p in paths_to_check) {
    if (file.exists(p)) {
      found_path <- p
      break
    }
  }

  if (is.null(found_path)) {
    stop(paste("CRITICAL ERROR: Could not find file:", filename))
  }

  con <- gzfile(found_path, "rb")
  on.exit(close(con))
  magic <- readBin(con, integer(), n = 1, endian = "big")

  if (magic == 2051) { # Images
    num <- readBin(con, integer(), n = 1, endian = "big")
    rows <- readBin(con, integer(), n = 1, endian = "big")
    cols <- readBin(con, integer(), n = 1, endian = "big")
    pixels <- readBin(con, integer(), n = num * rows * cols, size = 1, signed = FALSE)
    return(matrix(pixels, nrow = num, ncol = rows * cols, byrow = TRUE) / 255)
  } else if (magic == 2049) { # Labels
    num <- readBin(con, integer(), n = 1, endian = "big")
    return(readBin(con, integer(), n = num, size = 1, signed = FALSE))
  } else {
    stop("Unknown Magic Number")
  }
}

one_hot_encode <- function(y, classes = 10) {
  n <- length(y)
  y_enc <- matrix(0, nrow = classes, ncol = n)
  y_enc[cbind(y + 1, 1:n)] <- 1
  return(y_enc)
}

# ==============================================================================
# PART 1: ACTIVATIONS & LOSS
# ==============================================================================

relu <- function(x) { ifelse(x < 0, 0, x) }

logsoftmax <- function(x) {
  x_max <- apply(x, 2, max)
  x_shifted <- sweep(x, 2, x_max, "-")
  lse <- log(colSums(exp(x_shifted)))
  return(sweep(x_shifted, 2, lse, "-"))
}

cross_entropy_loss <- function(Y, logprobs) {
  m <- ncol(logprobs)
  loss <- -sum(Y * logprobs) / m
  return(loss)
}

# ==============================================================================
# PART 2: CONVOLUTION LAYER
# ==============================================================================

conv2d_forward <- function(X, W, stride=1, padding=1) {
  N <- dim(X)[1]; H_in <- dim(X)[2]; W_in <- dim(X)[3]; C_in <- dim(X)[4]
  K_h <- dim(W)[1]; K_w <- dim(W)[2]; C_out <- dim(W)[4]

  H_out <- floor((H_in - K_h + 2*padding) / stride) + 1
  W_out <- floor((W_in - K_w + 2*padding) / stride) + 1
  Z <- array(0, dim = c(N, H_out, W_out, C_out))

  if (padding > 0) {
    X_pad <- array(0, dim = c(N, H_in + 2*padding, W_in + 2*padding, C_in))
    X_pad[, (padding+1):(padding+H_in), (padding+1):(padding+W_in), ] <- X
  } else { X_pad <- X }

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
  N <- dim(X)[1]; H_in <- dim(X)[2]; W_in <- dim(X)[3]; C_in <- dim(X)[4]
  K_h <- dim(W)[1]; K_w <- dim(W)[2]; C_out <- dim(W)[4]
  H_out <- dim(dZ)[2]; W_out <- dim(dZ)[3]

  dW <- array(0, dim = dim(W))
  dX_padded <- array(0, dim = c(N, H_in + 2*padding, W_in + 2*padding, C_in))

  if (padding > 0) {
    X_pad <- array(0, dim = c(N, H_in + 2*padding, W_in + 2*padding, C_in))
    X_pad[, (padding+1):(padding+H_in), (padding+1):(padding+W_in), ] <- X
  } else { X_pad <- X }

  for (n in 1:N) {
    for (d in 1:C_out) {
      for (i in 1:H_out) {
        for (j in 1:W_out) {
          current_grad <- dZ[n, i, j, d]
          if (current_grad == 0) next

          h_start <- (i-1)*stride + 1; h_end <- h_start + K_h - 1
          w_start <- (j-1)*stride + 1; w_end <- w_start + K_w - 1

          patch <- X_pad[n, h_start:h_end, w_start:w_end, ]
          dW[, , , d] <- dW[, , , d] + (patch * current_grad)

          filter <- W[, , , d]
          dX_padded[n, h_start:h_end, w_start:w_end, ] <-
            dX_padded[n, h_start:h_end, w_start:w_end, ] + (filter * current_grad)
        }
      }
    }
  }

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

          mask <- matrix(0, nrow=kernel_size, ncol=kernel_size)
          mask[max_idx] <- 1

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
  out <- matrix(as.vector(X), nrow=N, byrow=FALSE)
  return(list(out=out, original_dims=dims))
}

flatten_backward <- function(dZ, original_dims) {
  return(array(dZ, dim=original_dims))
}

# ==============================================================================
# PART 5: VISUALIZATION UTILS
# ==============================================================================

view_digit <- function(index, X, Y) {
  label_map <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

  # Extract image [28, 28]
  img_mat <- X[index, , , 1]

  # Fix Orientation: Reverse Columns (Flip vertically)
  img_mat_viz <- img_mat[, 28:1]

  label_idx <- which.max(Y[, index]) - 1
  label_name <- label_map[label_idx + 1]

  image(1:28, 1:28, img_mat_viz,
        col = gray((0:255)/255),
        axes = FALSE,
        main = paste0(label_name, " (", label_idx, ")"))
}

# ==============================================================================
# PART 6: OPTIMIZER (ADAM)
# ==============================================================================

init_adam <- function(params) {
  m <- list()
  v <- list()
  for (name in names(params)) {
    m[[name]] <- array(0, dim = dim(params[[name]]))
    v[[name]] <- array(0, dim = dim(params[[name]]))
  }
  return(list(m=m, v=v))
}

update_adam <- function(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8) {
  # 1. Update Momentum
  m_next <- beta1 * m + (1 - beta1) * grad

  # 2. Update Velocity (Squared Gradients)
  v_next <- beta2 * v + (1 - beta2) * (grad^2)

  # 3. Bias Correction
  m_hat <- m_next / (1 - beta1^t)
  v_hat <- v_next / (1 - beta2^t)

  # 4. Update Parameter
  param_next <- param - lr * (m_hat / (sqrt(v_hat) + epsilon))

  return(list(param=param_next, m=m_next, v=v_next))
}
