# Manual Backpropagation (Single feature, single layer, no activation function)
# Init params
W = 0.1
b = 0
lr = 0.01
n = length(Y)

# Normalize
X <- scale(mtcars$hp)
Y <- scale(mtcars$mpg)

history <- numeric(1000)

# Training loop ----------------------------------------------------------------
for(i in 1:100){
  # Forward
  Out = X * W + b
  # Calculate Loss
  Loss = (1/2) * sum((Out-Y)^2) / n
  history[i] = Loss

  # Graident calculation
  dLoss = Out - Y
  dW = sum(dLoss * X) / n
  db = sum(dLoss) / n

  # Update params
  W = W - dW * lr
  b = b - db * lr

  # Print the loss every 10 steps
  if (i %% 10 == 0) cat("Epoch:", i, "Loss:", Loss, "\n")
}
# Sonuç
print(paste("W:", W))
print(paste("b:", b))

# Visualize --------------------------------------------------------------------
par(mfrow = c(1, 2))

# 1. Loss Graph
plot(history, type = "l", col = "blue", lwd = 2,
     main = "Training Loss", xlab = "Epoch", ylab = "MSE")
grid()

# 2. Regression line
plot(X, Y, pch = 19, col = "darkblue",
     main = "Regression Fit", xlab = "Normalized HP", ylab = "Normalized MPG")
abline(a = b, b = W, col = "darkred", lwd = 2)

par(mfrow = c(1, 1))
