library(torch)
library(torchvision)
library(coro)

# 1. Data Preparation ----------------------------------------------------------

train_transforms <- function(img) {
  img <- transform_to_tensor(img)
  img <- transform_normalize(img, mean = 0.5, std = 0.5)
  return(img)
}

ds_train <- fashion_mnist_dataset(root = "./data", train = TRUE, download = TRUE, transform = train_transforms)
ds_val   <- fashion_mnist_dataset(root = "./data", train = FALSE, download = TRUE, transform = train_transforms)

dl_train <- dataloader(ds_train, batch_size = 64, shuffle = TRUE)
dl_val   <- dataloader(ds_val, batch_size = 64, shuffle = FALSE)

# 2. Architecture Components ---------------------------------------------------

# Single Layer within a Dense Block
# Structure: BN -> ReLU -> Conv(3x3)
# (Simplified from full bottleneck version for CPU speed on 28x28 images)
conv_block <- nn_module(
  "ConvBlock",
  initialize = function(in_channels, growth_rate) {
    self$bn <- nn_batch_norm2d(in_channels)
    self$relu <- nn_relu(inplace = TRUE)
    self$conv <- nn_conv2d(in_channels, growth_rate, kernel_size = 3, padding = 1, bias = FALSE)
  },
  forward = function(x) {
    out <- self$conv(self$relu(self$bn(x)))
    # DenseNet Magic: Concatenate input with output
    return(torch_cat(list(x, out), dim = 2))
  }
)

# Dense Block
# A stack of 'num_layers' conv_blocks
dense_block <- nn_module(
  "DenseBlock",
  initialize = function(num_layers, in_channels, growth_rate) {
    layers <- list()
    for (i in 1:num_layers) {
      # Input to layer i has (in_channels + (i-1)*growth_rate) channels
      layers[[i]] <- conv_block(in_channels + (i - 1) * growth_rate, growth_rate)
    }
    # FIX: Use do.call to pass list elements as arguments
    # unlist() breaks R6 objects in this context
    self$net <- do.call(nn_sequential, layers)
  },
  forward = function(x) {
    return(self$net(x))
  }
)

# Transition Layer
# Reduces channels (1x1 conv) and spatial dimensions (AvgPool)
transition_block <- nn_module(
  "TransitionBlock",
  initialize = function(in_channels, out_channels) {
    self$net <- nn_sequential(
      nn_batch_norm2d(in_channels),
      nn_relu(inplace = TRUE),
      nn_conv2d(in_channels, out_channels, kernel_size = 1, bias = FALSE),
      nn_avg_pool2d(kernel_size = 2, stride = 2)
    )
  },
  forward = function(x) {
    return(self$net(x))
  }
)

# 3. Main Architecture: Tiny DenseNet ------------------------------------------

TinyDenseNet <- nn_module(
  "TinyDenseNet",
  initialize = function(growth_rate = 12, num_classes = 10) {

    # Initial Convolution (28x28 -> 28x28)
    num_channels <- 24 # Starting channels
    self$features <- nn_sequential(
      nn_conv2d(1, num_channels, kernel_size = 3, padding = 1, bias = FALSE)
    )

    # --- Block 1 ---
    num_layers <- 4
    self$block1 <- dense_block(num_layers, num_channels, growth_rate)
    num_channels <- num_channels + num_layers * growth_rate

    # Transition 1
    self$trans1 <- transition_block(num_channels, floor(num_channels * 0.5))
    num_channels <- floor(num_channels * 0.5)

    # --- Block 2 ---
    self$block2 <- dense_block(num_layers, num_channels, growth_rate)
    num_channels <- num_channels + num_layers * growth_rate

    # Transition 2
    self$trans2 <- transition_block(num_channels, floor(num_channels * 0.5))
    num_channels <- floor(num_channels * 0.5)

    # --- Block 3 ---
    self$block3 <- dense_block(num_layers, num_channels, growth_rate)
    num_channels <- num_channels + num_layers * growth_rate

    # Final Batch Norm & Classifier
    self$bn_final <- nn_batch_norm2d(num_channels)
    self$relu_final <- nn_relu(inplace = TRUE)
    self$avgpool <- nn_adaptive_avg_pool2d(c(1, 1))
    self$fc <- nn_linear(num_channels, num_classes)
  },

  forward = function(x) {
    out <- self$features(x)
    out <- self$block1(out)
    out <- self$trans1(out)
    out <- self$block2(out)
    out <- self$trans2(out)
    out <- self$block3(out)

    out <- self$relu_final(self$bn_final(out))
    out <- self$avgpool(out)
    out <- torch_flatten(out, start_dim = 2)
    out <- self$fc(out)
    return(out)
  }
)

# 4. Training Setup ------------------------------------------------------------

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Training on:", device$type, "\n")

model <- TinyDenseNet(growth_rate = 12, num_classes = 10)
model$to(device = device)

# Initialization
for (p in model$parameters) {
  if (length(p$shape) > 1) nn_init_kaiming_normal_(p, mode = "fan_out", nonlinearity = "relu")
}

criterion <- nn_cross_entropy_loss()
# Using Adam as DenseNets benefit from adaptive rates, keeping decay for regularization
optimizer <- optim_adam(model$parameters, lr = 0.001, weight_decay = 1e-4)

# History
num_epochs <- 10
history_batch <- data.frame(step=integer(), epoch=integer(), train_loss=numeric(), train_acc=numeric())
history_epoch <- data.frame(step=integer(), epoch=integer(), val_loss=numeric(), val_acc=numeric())

global_step <- 0
log_interval <- 50

# 5. Training Loop -------------------------------------------------------------

cat("Starting DenseNet Training...\n")

for (epoch in 1:num_epochs) {
  model$train()

  interval_loss <- 0; interval_correct <- 0; interval_total <- 0
  batch_idx <- 0

  coro::loop(for (b in dl_train) {
    batch_idx <- batch_idx + 1
    global_step <- global_step + 1

    step <- local({
      x <- b$x$to(device = device)
      y <- b$y$to(device = device)
      if (length(x$shape) == 3) x <- x$unsqueeze(2)

      optimizer$zero_grad()
      output <- model(x)
      loss <- criterion(output, y)
      loss$backward()
      # Clipping isn't strictly necessary for DenseNet usually, but good for safety
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()

      preds <- torch_max(output, dim = 2)[[2]]
      correct <- (preds == y)$sum()$item()
      total <- y$size(1)
      list(loss = loss$item(), correct = correct, total = total)
    })

    interval_loss <- interval_loss + step$loss
    interval_correct <- interval_correct + step$correct
    interval_total <- interval_total + step$total

    if (batch_idx %% log_interval == 0) {
      log_loss <- interval_loss / log_interval
      log_acc <- interval_correct / interval_total

      history_batch <- rbind(history_batch, data.frame(
        step = global_step, epoch = epoch, train_loss = log_loss, train_acc = log_acc
      ))

      cat(sprintf("\rEpoch %d | Batch %d | Loss: %.4f | Acc: %.2f%%",
                  epoch, batch_idx, log_loss, log_acc * 100))
      flush.console()
      interval_loss <- 0; interval_correct <- 0; interval_total <- 0
    }
  })

  # Validation
  model$eval()
  val_loss_sum <- 0; val_correct <- 0; val_total <- 0

  with_no_grad({
    coro::loop(for (b in dl_val) {
      step <- local({
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
      val_loss_sum <- val_loss_sum + step$loss
      val_correct <- val_correct + step$correct
      val_total <- val_total + step$total
    })
  })

  avg_val_loss <- val_loss_sum / length(dl_val)
  avg_val_acc  <- val_correct / val_total

  history_epoch <- rbind(history_epoch, data.frame(
    step = global_step, epoch = epoch, val_loss = avg_val_loss, val_acc = avg_val_acc
  ))

  cat(sprintf("\n>> Epoch %d Finished | Val Loss: %.4f (Acc: %.2f%%)\n",
              epoch, avg_val_loss, avg_val_acc*100))
}

# 6. Visualization -------------------------------------------------------------

par(mfrow=c(1, 2))

y_range_loss <- range(c(history_batch$train_loss, history_epoch$val_loss))
plot(history_batch$step, history_batch$train_loss, type="l", col="blue", lwd=1, ylim=y_range_loss,
     xlab="Global Step", ylab="Loss", main="DenseNet Loss")
points(history_epoch$step, history_epoch$val_loss, col="red", pch=19, cex=1.5)
lines(history_epoch$step, history_epoch$val_loss, col="red", lty=2)
legend("topright", legend=c("Train", "Val"), col=c("blue", "red"), lty=c(1, 2), pch=c(NA, 19))

y_range_acc <- range(c(history_batch$train_acc, history_epoch$val_acc))
plot(history_batch$step, history_batch$train_acc, type="l", col="blue", lwd=1, ylim=c(min(y_range_acc)-0.05, 1),
     xlab="Global Step", ylab="Accuracy", main="DenseNet Accuracy")
points(history_epoch$step, history_epoch$val_acc, col="red", pch=19, cex=1.5)
lines(history_epoch$step, history_epoch$val_acc, col="red", lty=2)
legend("bottomright", legend=c("Train", "Val"), col=c("blue", "red"), lty=c(1, 2), pch=c(NA, 19))
