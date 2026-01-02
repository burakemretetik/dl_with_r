library(torch)
library(torchvision)
library(coro)

# 1. Data Setup ---------------------------------------------------------------

train_transforms <- function(img) {
  img <- transform_to_tensor(img)
  # No resize needed, keeping 28x28 for speed
  img <- transform_normalize(img, mean = 0.5, std = 0.5)
  return(img)
}

ds_train_full <- fashion_mnist_dataset(root = "./data", train = TRUE, download = TRUE, transform = train_transforms)
ds_val_full   <- fashion_mnist_dataset(root = "./data", train = FALSE, download = TRUE, transform = train_transforms)

# Increased Batch Size to 128 for speed (Less overhead per epoch)
dl_train <- dataloader(ds_train_full, batch_size = 128, shuffle = TRUE)
dl_val   <- dataloader(ds_val_full, batch_size = 128, shuffle = FALSE)

# 2. NiN Architecture Components -----------------------------------------------

# Helper Function: Create a NiN Block
# A block consists of: Normal Conv -> 1x1 Conv -> 1x1 Conv
nin_block <- function(in_channels, out_channels, kernel_size, strides, padding) {
  nn_sequential(
    nn_conv2d(in_channels, out_channels, kernel_size, strides, padding),
    nn_relu(inplace = TRUE),
    nn_conv2d(out_channels, out_channels, kernel_size = 1), # 1x1 Conv
    nn_relu(inplace = TRUE),
    nn_conv2d(out_channels, out_channels, kernel_size = 1), # 1x1 Conv
    nn_relu(inplace = TRUE)
  )
}

NiN <- nn_module(
  "NiN",
  initialize = function(num_classes = 10) {
    self$net <- nn_sequential(

      # Block 1: Input [1, 28, 28] -> Output [32, 28, 28]
      nin_block(1, 32, kernel_size = 3, strides = 1, padding = 1),
      nn_max_pool2d(kernel_size = 3, stride = 2, padding = 1), # Downsample -> [32, 14, 14]
      nn_dropout(0.5),

      # Block 2: [32, 14, 14] -> [64, 14, 14]
      nin_block(32, 64, kernel_size = 3, strides = 1, padding = 1),
      nn_max_pool2d(kernel_size = 3, stride = 2, padding = 1), # Downsample -> [64, 7, 7]
      nn_dropout(0.5),

      # Block 3: [64, 7, 7] -> [num_classes, 7, 7]
      # Notice: The last block outputs exactly 'num_classes' channels (10)
      nin_block(64, num_classes, kernel_size = 3, strides = 1, padding = 1),

      # Global Average Pooling Replacement
      # We reduce [10, 7, 7] to [10, 1, 1] by taking the average of each map
      nn_adaptive_avg_pool2d(c(1, 1)),

      nn_flatten() # [Batch, 10, 1, 1] -> [Batch, 10]
    )
  },

  forward = function(x) {
    return(self$net(x))
  }
)

# 3. Training Setup ------------------------------------------------------------

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Training on:", device$type, "\n")

model <- NiN(num_classes = 10)
model$to(device = device)

# Initialization (Xavier is crucial for 1x1 convs)
for (p in model$parameters) {
  if (length(p$shape) > 1) nn_init_xavier_uniform_(p)
}

criterion <- nn_cross_entropy_loss()
optimizer <- optim_adam(model$parameters, lr = 0.001)

# History storage
# Increased epochs for smoother plots
num_epochs <- 15
history <- data.frame(
  epoch = 1:num_epochs,
  train_loss = numeric(num_epochs),
  val_loss = numeric(num_epochs),
  train_acc = numeric(num_epochs),
  val_acc = numeric(num_epochs)
)

# 4. Training Loop -------------------------------------------------------------

cat("Starting training loop...\n")

for (epoch in 1:num_epochs) {

  # --- TRAIN STEP ---
  model$train()
  train_loss_sum <- 0
  train_correct <- 0
  train_total <- 0
  batch_idx <- 0

  coro::loop(for (b in dl_train) {
    batch_idx <- batch_idx + 1

    step_res <- local({
      x <- b$x$to(device = device)
      y <- b$y$to(device = device)

      if (length(x$shape) == 3) x <- x$unsqueeze(2)

      optimizer$zero_grad()
      output <- model(x)
      loss <- criterion(output, y)
      loss$backward()
      optimizer$step()

      preds <- torch_max(output, dim = 2)[[2]]
      correct <- (preds == y)$sum()$item()
      total <- y$size(1)

      list(loss = loss$item(), correct = correct, total = total)
    })

    train_loss_sum <- train_loss_sum + step_res$loss
    train_correct <- train_correct + step_res$correct
    train_total <- train_total + step_res$total

    # Less frequent printing since batch size is larger (steps are fewer)
    if (batch_idx %% 20 == 0) {
      cat(sprintf("Epoch %d | Batch %d | Loss: %.4f\r", epoch, batch_idx, step_res$loss))
      flush.console()
    }
  })

  cat("\n")
  avg_train_loss <- train_loss_sum / length(dl_train)
  avg_train_acc <- train_correct / train_total

  # --- VALIDATION STEP ---
  model$eval()
  val_loss_sum <- 0
  val_correct <- 0
  val_total <- 0

  with_no_grad({
    coro::loop(for (b in dl_val) {
      step_res <- local({
        x <- b$x$to(device = device)
        y <- b$y$to(device = device)

        if (length(x$shape) == 3) x <- x$unsqueeze(2)

        output <- model(x)
        loss <- criterion(output, y)

        preds <- torch_max(output, dim = 2)[[2]]
        correct <- (preds == y)$sum()$item()
        total <- y$size(1)

        list(loss = loss$item(), correct = correct, total = total)
      })
      val_loss_sum <- val_loss_sum + step_res$loss
      val_correct <- val_correct + step_res$correct
      val_total <- val_total + step_res$total
    })
  })

  avg_val_loss <- val_loss_sum / length(dl_val)
  avg_val_acc <- val_correct / val_total

  history$train_loss[epoch] <- avg_train_loss
  history$val_loss[epoch]   <- avg_val_loss
  history$train_acc[epoch]  <- avg_train_acc
  history$val_acc[epoch]    <- avg_val_acc

  # Expanded print statement to show Val Loss and Accuracy clearly
  cat(sprintf(">> Epoch %d | Train Loss: %.4f (Acc: %.2f%%) | Val Loss: %.4f (Acc: %.2f%%)\n",
              epoch, avg_train_loss, avg_train_acc * 100, avg_val_loss, avg_val_acc * 100))
}

# 5. Plotting ------------------------------------------------------------------

par(mfrow = c(1, 2))
# Plot Loss
y_loss_range <- range(c(history$train_loss, history$val_loss))
plot(history$epoch, history$train_loss, type = "o", col = "blue",
     ylim = y_loss_range,
     xlab = "Epoch", ylab = "Loss", main = "NiN Loss Curve", lwd = 2)
lines(history$epoch, history$val_loss, type = "o", col = "red", lwd = 2)
legend("topright", legend = c("Train", "Val"), col = c("blue", "red"), lty = 1, lwd = 2, cex = 0.8)

# Plot Accuracy
y_acc_range <- range(c(history$train_acc, history$val_acc))
plot(history$epoch, history$train_acc, type = "o", col = "blue",
     ylim = c(min(y_acc_range) - 0.05, 1.0), # Adaptive limits for better visibility
     xlab = "Epoch", ylab = "Accuracy", main = "NiN Accuracy Curve", lwd = 2)
lines(history$epoch, history$val_acc, type = "o", col = "red", lwd = 2)
legend("bottomright", legend = c("Train", "Val"), col = c("blue", "red"), lty = 1, lwd = 2, cex = 0.8)
par(mfrow = c(1, 1))
