# Deep Learning with R

Team project by [@burakemretetik](https://github.com/burakemretetik) and [@ckrkc](https://github.com/ckrkc) (Dec 2025 – Jan 2026).

Classic deep learning architectures implemented with [torch for R](https://torch.mlverse.org/) and trained on Fashion-MNIST, plus a character level RNN.

| Script | Model |
|---|---|
| `cnn.R`, `cnn_lib.R` | basic CNN, SGD and Adam optimizers |
| `lenet.R` | LeNet with batch norm, SGD with momentum |
| `tiny_alexnet.R` | smaller AlexNet |
| `nin.R` | Network in Network |
| `tiny_googlenet.R` | small GoogLeNet with inception blocks |
| `tiny_resnet.R` | small ResNet |
| `tiny_densenet.R` | small DenseNet |
| `rnn.R`, `rnn_lib.R` | character level RNN on *The Time Machine* (`timemachine.txt`) |

Requirements: `torch`, `torchvision`, `coro`, `stringr`. Fashion-MNIST is downloaded to `./data` on first run.
