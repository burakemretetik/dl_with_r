# General utils ----------------------------------------------------------------

relu = function(x){
  ifelse(x<0, 0, x)
}

relu_deriv = function(x){
  ifelse(x>0, 1, 0)
}

softmax <- function(x) {
  # Shift by max to prevent overflow
  s <- x - apply(x, 2, max)
  e <- exp(s)
  return(e / colSums(e))
}

logsoftmax = function(x){
  x_max = apply(x, 2, max)
  x_shifted = sweep(x, 2, x_max, "-") # x - x(max)
  lse = log(colSums(exp(x_shifted))) # lse = log(sum(e^(x - x(max))))
  return(sweep(x_shifted, 2, lse)) # x - x(max) - lse
}

cross_entropy_loss = function(Y, logprobs){
  m = ncol(logprobs)
  loss = -sum(Y * logprobs)/m
  return(loss)
}

#Fashion MNIST Data ------------------------------------------------------------
read_idx_file <- function(filename) {
  if (!file.exists(filename)) stop("File not found: ", filename)
  con <- gzfile(filename, "rb")
  on.exit(close(con))
  magic <- readBin(con, integer(), n = 1, endian = "big")

  if (magic == 2051) { # Images
    num_images <- readBin(con, integer(), n = 1, endian = "big")
    rows <- readBin(con, integer(), n = 1, endian = "big")
    cols <- readBin(con, integer(), n = 1, endian = "big")
    pixels <- readBin(con, integer(), n = num_images * rows * cols, size = 1, signed = FALSE)
    return(matrix(pixels, nrow = num_images, ncol = rows * cols, byrow = TRUE) / 255)
  } else if (magic == 2049) { # Labels
    num_items <- readBin(con, integer(), n = 1, endian = "big")
    return(readBin(con, integer(), n = num_items, size = 1, signed = FALSE))
  } else {
    stop("Unknown Magic Number: ", magic)
  }
}

one_hot_encode <- function(y, classes = 10) {
  n_samples <- length(y)
  y_encoded <- matrix(0, nrow = classes, ncol = n_samples)
  indices <- cbind(y + 1, 1:n_samples)
  y_encoded[indices] <- 1
  return(y_encoded)
}

# View an image in the fashion mnist dataset
view_image <- function(index, X, Y) {
  # Fashion MNIST Class Mapping
  label_map <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

  # 1. Extract the specific image
  # Input X is expected to be [N, 28, 28, 1]
  # We extract the 28x28 matrix directly
  img_mat <- X[index, , , 1]

  # 2. Orient for R plotting
  # R image() plots from bottom-left, so we transpose and reverse columns
  # to make it look like a proper image
  img_mat_viz <- t(apply(img_mat, 2, rev))

  # 3. Decode One-Hot Label
  # Y is [10, N], so we look at the column
  label_idx <- which.max(Y[, index]) - 1 # Convert 1-10 to 0-9
  label_name <- label_map[label_idx + 1]

  # 4. Plot
  image(1:28, 1:28, img_mat_viz,
        col = gray((0:255)/255),
        axes = FALSE,
        main = paste0(label_name, " (", label_idx, ")"))
}
