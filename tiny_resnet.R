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

# Keeping Batch Size 128 for stability and speed
dl_train <- dataloader(ds_train, batch_size = 128, shuffle = TRUE)
dl_val   <- dataloader(ds_val, batch_size = 128, shuffle = FALSE)

# 2. Architecture Components: Residual Block -----------------------------------

ResidualBlock <- nn_module(
  "ResidualBlock",

  initialize = function(in_channels, out_channels, stride = 1) {
    self$conv1 <- nn_conv2d(in_channels, out_channels, kernel_size = 3,
                            stride = stride, padding = 1, bias = FALSE)
    self$bn1   <- nn_batch_norm2d(out_channels)
    self$relu  <- nn_relu(inplace = TRUE)

    self$conv2 <- nn_conv2d(out_channels, out_channels, kernel_size = 3,
                            stride = 1, padding = 1, bias = FALSE)
    self$bn2   <- nn_batch_norm2d(out_channels)

    self$shortcut <- nn_sequential()
    if (stride != 1 || in_channels != out_channels) {
      self$shortcut <- nn_sequential(
        nn_conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = FALSE),
        nn_batch_norm2d(out_channels)
      )
    }
  },

  forward = function(x) {
    out <- self$conv1(x)
    out <- self$bn1(out)
    out <- self$relu(out)
    out <- self$conv2(out)
    out <- self$bn2(out)
    out <- out + self$shortcut(x)
    out <- self$relu(out)
    return(out)
  }
)

# 3. Main Architecture: Tiny ResNet --------------------------------------------

TinyResNet <- nn_module(
  "TinyResNet",

  initialize = function(num_classes = 10) {
    self$in_channels <- 16

    self$conv1 <- nn_conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1, bias = FALSE)
    self$bn1   <- nn_batch_norm2d(16)
    self$relu  <- nn_relu(inplace = TRUE)

    make_layer <- function(out_channels, num_blocks, stride) {
      layers <- list()
      layers[[1]] <- ResidualBlock(self$in_channels, out_channels, stride)
      self$in_channels <- out_channels
      if (num_blocks > 1) {
        for (i in 2:num_blocks) {
          layers[[i]] <- ResidualBlock(out_channels, out_channels, stride = 1)
        }
      }
      return(do.call(nn_sequential, layers))
    }

    self$layer1 <- make_layer(out_channels = 16, num_blocks = 2, stride = 1)
    self$layer2 <- make_layer(out_channels = 32, num_blocks = 2, stride = 2)
    self$layer3 <- make_layer(out_channels = 64, num_blocks = 2, stride = 2)

    self$avgpool <- nn_adaptive_avg_pool2d(c(1, 1))
    self$fc      <- nn_linear(64, num_classes)
  },

  forward = function(x) {
    x <- self$conv1(x)
    x <- self$bn1(x)
    x <- self$relu(x)
    x <- self$layer1(x)
    x <- self$layer2(x)
    x <- self$layer3(x)
    x <- self$avgpool(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fc(x)
    return(x)
  }
)

# 4. Training Setup ------------------------------------------------------------

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Training on:", device$type, "\n")

model <- TinyResNet(num_classes = 10)
model$to(device = device)

for (p in model$parameters) {
  if (length(p$shape) > 1) nn_init_kaiming_normal_(p, mode = "fan_out", nonlinearity = "relu")
}

criterion <- nn_cross_entropy_loss()

# FIX: Switched to Adam.
# LR 0.001 is standard for Adam. It adapts automatically.
optimizer <- optim_adam(model$parameters, lr = 0.001, weight_decay = 1e-4)

# History
num_epochs <- 10
history_batch <- data.frame(step=integer(), epoch=integer(), train_loss=numeric(), train_acc=numeric())
history_epoch <- data.frame(step=integer(), epoch=integer(), val_loss=numeric(), val_acc=numeric())

global_step <- 0
log_interval <- 20

cat("Starting ResNet Training (Adam + Clipping)...\n")

for (epoch in 1:num_epochs) {
  model$train()

  interval_loss <- 0
  interval_correct <- 0
  interval_total <- 0
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

      # KEEPING CLIPPING: This is our safety net against explosions
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

  # --- VALIDATION ---
  model$eval()
  val_loss_sum <- 0
  val_correct <- 0
  val_total <- 0

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
     xlab="Global Step", ylab="Loss", main="Loss: Train(Line) vs Val(Points)")
points(history_epoch$step, history_epoch$val_loss, col="red", pch=19, cex=1.5)
lines(history_epoch$step, history_epoch$val_loss, col="red", lty=2)
legend("topright", legend=c("Train", "Val"), col=c("blue", "red"), lty=c(1, 2), pch=c(NA, 19))

y_range_acc <- range(c(history_batch$train_acc, history_epoch$val_acc))
plot(history_batch$step, history_batch$train_acc, type="l", col="blue", lwd=1, ylim=c(min(y_range_acc)-0.05, 1),
     xlab="Global Step", ylab="Accuracy", main="Acc: Train(Line) vs Val(Points)")
points(history_epoch$step, history_epoch$val_acc, col="red", pch=19, cex=1.5)
lines(history_epoch$step, history_epoch$val_acc, col="red", lty=2)
legend("bottomright", legend=c("Train", "Val"), col=c("blue", "red"), lty=c(1, 2), pch=c(NA, 19))
