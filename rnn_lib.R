library(torch)
library(stringr)

# PART 0: DATA LOADING ---------------------------------------------------------

get_time_machine_data <- function() {
  url <- "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
  filepath <- "timemachine.txt"

  if (!file.exists(filepath)) {
    download.file(url, filepath)
  }

  lines <- readLines(filepath, warn = FALSE)

  # Processing: Lowercase, remove non-alphabets, collapse to char stream
  text <- paste(lines, collapse = " ")
  text <- tolower(text)
  text <- gsub("[^a-z]+", " ", text)
  text <- str_squish(text)

  # Split into characters
  chars <- strsplit(text, "")[[1]]
  return(chars)
}

# PART 1: VOCABULARY UTILS -----------------------------------------------------

build_vocab <- function(tokens) {
  unique_tokens <- sort(unique(tokens))

  # Add <unk> token at index 1
  idx_to_token <- c("<unk>", unique_tokens)
  token_to_idx <- setNames(1:length(idx_to_token), idx_to_token)

  # Closure for lookups
  to_indices <- function(char_seq) {
    idx <- unname(token_to_idx[char_seq])
    idx[is.na(idx)] <- token_to_idx["<unk>"]
    return(idx)
  }

  to_tokens <- function(idx_seq) {
    # FIX: Use 'inherits' instead of 'is_torch_tensor' for robustness
    if (inherits(idx_seq, "torch_tensor")) idx_seq <- as.numeric(idx_seq)
    return(idx_to_token[idx_seq])
  }

  return(list(
    size = length(idx_to_token),
    to_indices = to_indices,
    to_tokens = to_tokens
  ))
}

# PART 2: DATA ITERATOR --------------------------------------------------------

# Random sampling data iterator
data_iter_random <- function(corpus, batch_size, num_steps) {
  # Random offset to vary sequence start
  corpus <- corpus[sample(1:5, 1):length(corpus)]

  # Calculate number of full subsequences
  num_subseqs <- (length(corpus) - 1) %/% num_steps
  initial_indices <- seq(1, num_subseqs * num_steps, by = num_steps)

  # Shuffle indices for random sampling
  initial_indices <- sample(initial_indices)

  # Filter to fit full batches
  num_batches <- length(initial_indices) %/% batch_size
  initial_indices <- initial_indices[1:(num_batches * batch_size)]

  # Reshape indices into [Batch, Batch_Size]
  initial_indices <- matrix(initial_indices, nrow = batch_size, byrow = TRUE)

  # Return generator function
  batch_idx <- 1
  iterator <- function() {
    if (batch_idx > ncol(initial_indices)) return(NULL)

    start_indices <- initial_indices[, batch_idx]
    batch_idx <<- batch_idx + 1

    X_list <- list()
    Y_list <- list()

    for (i in 1:batch_size) {
      idx <- start_indices[i]
      X_list[[i]] <- corpus[idx:(idx + num_steps - 1)]
      Y_list[[i]] <- corpus[(idx + 1):(idx + num_steps)]
    }

    # Convert to tensors [Batch, Steps]
    X <- torch_tensor(do.call(rbind, X_list), dtype = torch_long())
    Y <- torch_tensor(do.call(rbind, Y_list), dtype = torch_long())

    return(list(X=X, Y=Y))
  }

  return(iterator)
}

# PART 3: ENCODING -------------------------------------------------------------

# One-hot encoding for sequence data
# Input: [Batch, Steps] -> Output: [Steps, Batch, Vocab]
to_one_hot <- function(X, vocab_size) {
  # Transpose to [Steps, Batch] for time-major processing
  X_t <- torch_transpose(X, 1, 2)
  # Apply one-hot
  return(nnf_one_hot(X_t, num_classes = vocab_size)$to(dtype = torch_float32()))
}
