library(torch)
library(torchvision)
library(coro)

# 1. Data Setup (Subsets for Speed) --------------------------------------------

train_transforms <- function(img) {
  img <- transform_to_tensor(img)
  img <- transform_resize(img, size = c(224, 224))
  img <- transform_normalize(img, mean = 0.5, std = 0.5)
  return(img)
}

# Load full datasets
full_ds_train <- fashion_mnist_dataset(root = "./data", train = TRUE, download = TRUE, transform = train_transforms)
full_ds_val   <- fashion_mnist_dataset(root = "./data", train = FALSE, download = TRUE, transform = train_transforms)

# --- SPEED FIX: Subsetting both Train and Validation ---
# Train: 128 images (2 batches)
# Val:   64 images (1 batch)
ds_train_small <- dataset_subset(full_ds_train, indices = 1:128)
ds_val_small   <- dataset_subset(full_ds_val, indices = 1:64)

dl_train <- dataloader(ds_train_small, batch_size = 64, shuffle = TRUE)
dl_val   <- dataloader(ds_val_small, batch_size = 64, shuffle = FALSE)

# 2. Model Architecture --------------------------------------------------------

AlexNet <- nn_module(
  "AlexNet",
  initialize = function(num_classes = 10) {
    self$features <- nn_sequential(
      nn_conv2d(1, 64, kernel_size = 11, stride = 4, padding = 2),
      nn_batch_norm2d(64),
      nn_relu(inplace = TRUE),
      nn_max_pool2d(kernel_size = 3, stride = 2),

      nn_conv2d(64, 192, kernel_size = 5, padding = 2),
      nn_batch_norm2d(192),
      nn_relu(inplace = TRUE),
      nn_max_pool2d(kernel_size = 3, stride = 2),

      nn_conv2d(192, 384, kernel_size = 3, padding = 1),
      nn_batch_norm2d(384),
      nn_relu(inplace = TRUE),

      nn_conv2d(384, 256, kernel_size = 3, padding = 1),
      nn_batch_norm2d(256),
      nn_relu(inplace = TRUE),

      nn_conv2d(256, 256, kernel_size = 3, padding = 1),
      nn_batch_norm2d(256),
      nn_relu(inplace = TRUE),
      nn_max_pool2d(kernel_size = 3, stride = 2)
    )

    self$avgpool <- nn_adaptive_avg_pool2d(c(6, 6))

    self$classifier <- nn_sequential(
      nn_dropout(p = 0.5),
      nn_linear(256 * 6 * 6, 4096),
      nn_relu(inplace = TRUE),
      nn_dropout(p = 0.5),
      nn_linear(4096, 4096),
      nn_relu(inplace = TRUE),
      nn_linear(4096, num_classes)
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
model <- AlexNet(num_classes = 10)
model$to(device = device)

criterion <- nn_cross_entropy_loss()
optimizer <- optim_adam(model$parameters, lr = 0.0001)

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

  coro::loop(for (b in dl_train) {
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
  })

  avg_train_loss <- train_loss_sum / length(dl_train)

  # --- VALIDATION STEP ---
  model$eval() # Switch to evaluation mode
  val_loss_sum <- 0

  # specific context for no gradient calculation
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

  cat(sprintf("Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f\n",
              epoch, num_epochs, avg_train_loss, avg_val_loss))
}

# 5. Plotting ------------------------------------------------------------------

# Simple base R plot for Loss vs Epochs
plot(history$epoch, history$train_loss, type = "o", col = "blue",
     ylim = range(c(history$train_loss, history$val_loss)),
     xlab = "Epoch", ylab = "Loss", main = "Training vs Validation Loss",
     lwd = 2)

lines(history$epoch, history$val_loss, type = "o", col = "red", lwd = 2)

legend("topright", legend = c("Train Loss", "Val Loss"),
       col = c("blue", "red"), lty = 1, lwd = 2)
