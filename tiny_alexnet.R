library(torch)
library(torchvision)
library(coro)

# 1. Data Setup ----------------------------------------------------------------

train_transforms <- function(img) {
  img <- transform_to_tensor(img)
  # Keeping native 28x28 for speed
  img <- transform_normalize(img, mean = 0.5, std = 0.5)
  return(img)
}

ds_train_full <- fashion_mnist_dataset(root = "./data", train = TRUE, download = TRUE, transform = train_transforms)
ds_val_full   <- fashion_mnist_dataset(root = "./data", train = FALSE, download = TRUE, transform = train_transforms)

dl_train <- dataloader(ds_train_full, batch_size = 64, shuffle = TRUE)
dl_val   <- dataloader(ds_val_full, batch_size = 64, shuffle = FALSE)

# 2. Model Architecture (Slim AlexNet for CPU Speed) ---------------------------

AlexNet <- nn_module(
  "AlexNet",
  initialize = function(num_classes = 10) {
    self$features <- nn_sequential(
      # Reduced filters (16, 32, 64...) instead of (32, 64, 128...) for speed

      # Layer 1
      nn_conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1),
      nn_batch_norm2d(16),
      nn_relu(inplace = TRUE),
      nn_max_pool2d(kernel_size = 2, stride = 2),

      # Layer 2
      nn_conv2d(16, 32, kernel_size = 3, padding = 1),
      nn_batch_norm2d(32),
      nn_relu(inplace = TRUE),
      nn_max_pool2d(kernel_size = 2, stride = 2),

      # Layer 3
      nn_conv2d(32, 64, kernel_size = 3, padding = 1),
      nn_batch_norm2d(64),
      nn_relu(inplace = TRUE),

      # Layer 4
      nn_conv2d(64, 128, kernel_size = 3, padding = 1),
      nn_batch_norm2d(128),
      nn_relu(inplace = TRUE),

      # Layer 5
      nn_conv2d(128, 128, kernel_size = 3, padding = 1),
      nn_batch_norm2d(128),
      nn_relu(inplace = TRUE),
      nn_max_pool2d(kernel_size = 2, stride = 2)
    )

    self$avgpool <- nn_adaptive_avg_pool2d(c(3, 3))

    self$classifier <- nn_sequential(
      nn_dropout(p = 0.5),
      # Input dim: 128 channels * 3 * 3 = 1152
      nn_linear(128 * 3 * 3, 512),
      nn_relu(inplace = TRUE),
      nn_dropout(p = 0.5),
      nn_linear(512, 256),
      nn_relu(inplace = TRUE),
      nn_linear(256, num_classes)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
    return(x)
  }
)

# 3. Setup ---------------------------------------------------------------------

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Training on:", device$type, "\n")

model <- AlexNet(num_classes = 10)
model$to(device = device)

criterion <- nn_cross_entropy_loss()

# Using SGD with Momentum as requested (Lighter than Adam)
optimizer <- optim_sgd(model$parameters, lr = 0.01, momentum = 0.9)

# History storage
num_epochs <- 5
history <- data.frame(
  epoch = 1:num_epochs,
  train_loss = numeric(num_epochs),
  val_loss = numeric(num_epochs)
)

# 4. Training & Validation Loop ------------------------------------------------

cat("Starting training loop...\n")

for (epoch in 1:num_epochs) {

  # --- TRAIN STEP ---
  model$train()
  train_loss_sum <- 0
  batch_idx <- 0

  coro::loop(for (b in dl_train) {
    batch_idx <- batch_idx + 1

    loss <- local({
      x <- b$x$to(device = device)
      y <- b$y$to(device = device)

      if (length(x$shape) == 3) x <- x$unsqueeze(2)

      optimizer$zero_grad()
      output <- model(x)
      loss <- criterion(output, y)
      loss$backward()
      optimizer$step()
      loss$item()
    })
    train_loss_sum <- train_loss_sum + loss

    # Progress Check: Print every 50 batches
    if (batch_idx %% 50 == 0) {
      cat(sprintf("Epoch %d | Batch %d | Loss: %.4f\r", epoch, batch_idx, loss))
      flush.console()
    }
  })

  cat("\n") # New line after epoch finishes
  avg_train_loss <- train_loss_sum / length(dl_train)

  # --- VALIDATION STEP ---
  model$eval()
  val_loss_sum <- 0

  with_no_grad({
    coro::loop(for (b in dl_val) {
      loss <- local({
        x <- b$x$to(device = device)
        y <- b$y$to(device = device)

        if (length(x$shape) == 3) x <- x$unsqueeze(2)

        output <- model(x)
        loss <- criterion(output, y)
        loss$item()
      })
      val_loss_sum <- val_loss_sum + loss
    })
  })

  avg_val_loss <- val_loss_sum / length(dl_val)

  # --- STORE & LOG ---
  history$train_loss[epoch] <- avg_train_loss
  history$val_loss[epoch]   <- avg_val_loss

  cat(sprintf(">> Epoch %d Completed | Train Loss: %.4f | Val Loss: %.4f\n",
              epoch, avg_train_loss, avg_val_loss))
}

# 5. Plotting ------------------------------------------------------------------

plot(history$epoch, history$train_loss, type = "o", col = "blue",
     ylim = range(c(history$train_loss, history$val_loss)),
     xlab = "Epoch", ylab = "Loss", main = "Training vs Validation Loss (SGD)",
     lwd = 2)

lines(history$epoch, history$val_loss, type = "o", col = "red", lwd = 2)

legend("topright", legend = c("Train Loss", "Val Loss"),
       col = c("blue", "red"), lty = 1, lwd = 2)

# 6. Evaluation & Predictions --------------------------------------------------

cat("\n=== Calculating Final Accuracy & Visualizing ===\n")

# Set model to evaluation mode
model$eval()

class_names <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

correct_count <- 0
total_count <- 0
viz_batch <- NULL # To store one batch for plotting

# Iterate over Validation Set
with_no_grad({
  coro::loop(for (b in dl_val) {
    x <- b$x$to(device = device)
    y <- b$y$to(device = device)

    if (length(x$shape) == 3) x <- x$unsqueeze(2)

    outputs <- model(x)

    # Get predictions (torch_max returns list(values, indices))
    # Indices are 1-based class IDs in R Torch usually, but verify
    # fashion_mnist labels are 1-based integers in R torch
    preds <- torch_max(outputs, dim = 2)[[2]]

    total_count <- total_count + y$size(1)
    correct_count <- correct_count + (preds == y)$sum()$item()

    # Save the first batch for visualization later
    if (is.null(viz_batch)) {
      viz_batch <- list(
        x = x$to(device = "cpu"),
        y = y$to(device = "cpu"),
        preds = preds$to(device = "cpu")
      )
    }
  })
})

final_acc <- 100 * correct_count / total_count
cat(sprintf("Final Validation Accuracy: %.2f%%\n", final_acc))

# Visualize Predictions
# Plot 8 images in a grid
par(mfrow = c(2, 4), mar = c(2, 2, 4, 2))

for (i in 1:8) {
  # Extract image tensor (1, 28, 28) -> matrix (28, 28)
  img_tensor <- viz_batch$x[i, 1, , ]
  img_mat <- as.matrix(img_tensor)

  # Denormalize for plotting (Simple Min-Max scaling to 0-1)
  img_mat <- (img_mat - min(img_mat)) / (max(img_mat) - min(img_mat))

  # Rotate for R's image() function
  # R plots column 1 on X axis, so we transpose and reverse columns to orient upright
  img_mat <- t(apply(img_mat, 2, rev))

  true_label <- class_names[viz_batch$y[i]$item()]
  pred_label <- class_names[viz_batch$preds[i]$item()]

  # Color code title: Green if correct, Red if wrong
  col <- ifelse(true_label == pred_label, "forestgreen", "red")

  image(1:28, 1:28, img_mat, col = gray.colors(255), axes = FALSE,
        main = sprintf("True: %s\nPred: %s", true_label, pred_label),
        col.main = col)
}

# Reset layout
par(mfrow = c(1, 1))
