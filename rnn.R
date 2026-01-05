source("rnn_lib.R")

# 1. SETTINGS ------------------------------------------------------------------

BATCH_SIZE  <- 32
NUM_STEPS   <- 35   # Sequence length
NUM_HIDDENS <- 512  # Number of hidden neurons
LR          <- 1    # RNNs benefit from high LR with clipping
EPOCHS      <- 100

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Training Device:", device$type, "\n")

# Prepare Data
chars <- get_time_machine_data()
vocab <- build_vocab(chars)
corpus_indices <- vocab$to_indices(chars)
cat("Vocab Size:", vocab$size, "\n")

# Split Train/Validation (90% Train, 10% Val)
total_len <- length(corpus_indices)
train_len <- floor(total_len * 0.9)
train_corpus <- corpus_indices[1:train_len]
val_corpus   <- corpus_indices[(train_len + 1):total_len]

cat("Train Size:", length(train_corpus), "| Val Size:", length(val_corpus), "\n")

# 2. MODEL DEFINITIONS ---------------------------------------------------------

init_rnn_params <- function(vocab_size, num_hiddens, device) {
  num_inputs <- vocab_size
  num_outputs <- vocab_size

  # Helper for normal distribution initialization
  normal <- function(shape) {
    torch_randn(shape, device = device) * 0.01
  }

  # Hidden Layer Parameters
  W_xh <- normal(c(num_inputs, num_hiddens))
  W_hh <- normal(c(num_hiddens, num_hiddens))
  b_h <- torch_zeros(num_hiddens, device = device)

  # Output Layer Parameters
  W_hq <- normal(c(num_hiddens, num_outputs))
  b_q <- torch_zeros(num_outputs, device = device)

  # Enable gradients
  params <- list(W_xh, W_hh, b_h, W_hq, b_q)
  for (p in params) p$requires_grad_(TRUE)

  return(params)
}

init_rnn_state <- function(batch_size, num_hiddens, device) {
  # Initial hidden state (Zero matrix)
  return(list(torch_zeros(batch_size, num_hiddens, device = device)))
}

rnn_forward <- function(inputs, state, params) {
  # inputs: [Steps, Batch, Vocab] (One-hot)

  W_xh <- params[[1]]
  W_hh <- params[[2]]
  b_h <- params[[3]]
  W_hq <- params[[4]]
  b_q <- params[[5]]

  H <- state[[1]]
  outputs <- list()

  # Loop over time steps
  input_seq <- torch_unbind(inputs, dim = 1)

  for (X in input_seq) {
    # Equation: H = tanh(X @ W_xh + H @ W_hh + b_h)
    H <- torch_tanh(torch_mm(X, W_xh) + torch_mm(H, W_hh) + b_h)

    # Equation: Y = H @ W_hq + b_q
    Y <- torch_mm(H, W_hq) + b_q
    outputs <- c(outputs, list(Y))
  }

  # Concatenate outputs: [Steps * Batch, Vocab]
  return(list(
    output = torch_cat(outputs, dim = 1),
    state = list(H)
  ))
}

# 3. UTILS: PREDICTION & CLIPPING ----------------------------------------------

predict_rnn <- function(prefix, num_preds, vocab, params, device) {
  state <- init_rnn_state(1, NUM_HIDDENS, device)
  outputs <- c(vocab$to_indices(strsplit(prefix, "")[[1]]))

  # Helper: Convert last index to tensor input
  get_input <- function() {
    idx <- outputs[length(outputs)]
    # FIX: Added dtype = torch_long() because one_hot requires integers
    x <- torch_tensor(matrix(idx, ncol=1), device=device, dtype=torch_long())
    to_one_hot(x, vocab$size)
  }

  # Warm up state with prefix
  for (i in 1:(length(outputs)-1)) {
    # FIX: Added dtype = torch_long() here as well
    x <- torch_tensor(matrix(outputs[i], ncol=1), device=device, dtype=torch_long())
    x_oh <- to_one_hot(x, vocab$size)
    res <- rnn_forward(x_oh, state, params)
    state <- res$state
  }

  # Predict new characters
  for (i in 1:num_preds) {
    x_oh <- get_input()
    res <- rnn_forward(x_oh, state, params)

    y <- res$output
    next_idx <- as.numeric(torch_argmax(y, dim = 2))

    outputs <- c(outputs, next_idx)
    state <- res$state
  }

  return(paste(vocab$to_tokens(outputs), collapse = ""))
}

grad_clipping <- function(params, theta, device) {
  norm <- 0
  for (p in params) {
    norm <- norm + torch_sum(p$grad^2)
  }
  norm <- sqrt(norm)

  # Convert Tensor to Numeric for comparison
  if (as.numeric(norm) > theta) {
    for (p in params) {
      p$grad$mul_(theta / norm)
    }
  }
}

# 4. TRAINING LOOP -------------------------------------------------------------

params <- init_rnn_params(vocab$size, NUM_HIDDENS, device)
optimizer <- optim_sgd(params, lr = LR)
criterion <- nn_cross_entropy_loss()

# Metrics Storage
history <- data.frame(
  epoch = 1:EPOCHS,
  train_loss = numeric(EPOCHS),
  val_loss = numeric(EPOCHS),
  perplexity = numeric(EPOCHS)
)

cat("Starting Training...\n")

for (epoch in 1:EPOCHS) {

  # --- TRAIN STEP ---
  iter_train <- data_iter_random(train_corpus, BATCH_SIZE, NUM_STEPS)
  total_loss <- 0
  num_batches <- 0

  while (!is.null(batch <- iter_train())) {
    X <- batch$X$to(device = device)
    Y <- batch$Y$to(device = device)

    state <- init_rnn_state(BATCH_SIZE, NUM_HIDDENS, device)

    # Forward
    X_oh <- to_one_hot(X, vocab$size)
    res <- rnn_forward(X_oh, state, params)

    # Loss Calculation
    # Y: [Batch, Steps] -> Transpose [Steps, Batch] -> Flatten
    Y_flat <- torch_transpose(Y, 1, 2)$reshape(-1)

    loss <- criterion(res$output, Y_flat)

    # Backward
    optimizer$zero_grad()
    loss$backward()
    grad_clipping(params, 1, device)
    optimizer$step()

    total_loss <- total_loss + loss$item()
    num_batches <- num_batches + 1
  }

  avg_train_loss <- total_loss / num_batches
  perplexity <- exp(avg_train_loss)

  # --- VALIDATION STEP ---
  iter_val <- data_iter_random(val_corpus, BATCH_SIZE, NUM_STEPS)
  val_loss_sum <- 0
  val_batches <- 0

  # Use with_no_grad for validation to save memory/speed
  with_no_grad({
    while (!is.null(batch <- iter_val())) {
      X <- batch$X$to(device = device)
      Y <- batch$Y$to(device = device)

      state <- init_rnn_state(BATCH_SIZE, NUM_HIDDENS, device)

      X_oh <- to_one_hot(X, vocab$size)
      res <- rnn_forward(X_oh, state, params)

      Y_flat <- torch_transpose(Y, 1, 2)$reshape(-1)
      loss <- criterion(res$output, Y_flat)

      val_loss_sum <- val_loss_sum + loss$item()
      val_batches <- val_batches + 1
    }
  })

  avg_val_loss <- if (val_batches > 0) val_loss_sum / val_batches else NA

  # Store Metrics
  history$train_loss[epoch] <- avg_train_loss
  history$val_loss[epoch]   <- avg_val_loss
  history$perplexity[epoch] <- perplexity

  # Logging
  if (epoch %% 10 == 0 || epoch == 1) {
    cat(sprintf("Epoch %d | PPL: %.2f | Train Loss: %.4f | Val Loss: %.4f\n",
                epoch, perplexity, avg_train_loss, avg_val_loss))

    pred_text <- predict_rnn("time traveller", 50, vocab, params, device)
    cat("Pred:", pred_text, "\n")
  }
}

# 5. VISUALIZATION -------------------------------------------------------------

par(mfrow=c(1, 2))

# Plot 1: Loss
y_range <- range(c(history$train_loss, history$val_loss), na.rm=TRUE)
plot(history$epoch, history$train_loss, type="l", col="blue", lwd=2, ylim=y_range,
     xlab="Epoch", ylab="Loss", main="RNN Loss")
lines(history$epoch, history$val_loss, col="red", lwd=2)
legend("topright", legend=c("Train", "Val"), col=c("blue", "red"), lwd=2)

# Plot 2: Perplexity
plot(history$epoch, history$perplexity, type="l", col="purple", lwd=2,
     xlab="Epoch", ylab="Perplexity", main="Perplexity (Train)")
grid()

par(mfrow=c(1, 1))
