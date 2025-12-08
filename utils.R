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
