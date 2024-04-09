FL:
- centralized async -b 2048
- centralized_sync -b 128
- decent_async -le 6 -a 0.2 -b 256
- decent_sync -le 8 -b 256


| Dataset | Optimizer | Epochs | Learning Rate |
|:-:|:-:|:-:|:-:|
| Slicing5G | ? | ? | ? |
| IOT_DNL | Adam | 300 | ? |
| CIFAR | Adam | 100 | ? |
| FASHION | Adam | 50 | ? |
| MNIST | Adam | TF Default | TF Default |
| UNSW | Adam | 10 | 0.01 |