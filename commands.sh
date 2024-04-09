python3 single_training.py -d Slicing5G -o Adam -e 10 -lr 0.001 -b 64
python3 single_training.py -d IOT_DNL -o Adam -e 300 -lr 0.001 -b 64
python3 single_training.py -d CIFAR -o Adam -e 100 -lr 0.001 -b 64
python3 single_training.py -d FASHION -o Adam -e 50 -lr 0.001 -b 64
python3 single_training.py -d MNIST -o Adam -e 10 -lr 0.001 -b 64
python3 single_training.py -d UNSW -o Adam -e 10 -lr 0.01 -b 64

# -m 1 -b 2048
# -m 2 -b 128
# -m 3 -le 6 -a 0.2 -b 256
# -m 4 -le 8 -b 256

# Slicing5G
mpirun -np 9 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 9 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 9 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 9 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 5 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 5 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 5 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 5 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 3 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 3 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 3 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 3 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256

# IOT_DNL 
mpirun -np 9 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 9 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 9 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 9 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 5 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 5 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 5 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 5 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 3 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 3 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 3 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 3 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256

# CIFAR
mpirun -np 9 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 9 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 9 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 9 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 5 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 5 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 5 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 5 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 3 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 3 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 3 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 3 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256

# FASHION
mpirun -np 9 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 9 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 9 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 9 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 5 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 5 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 5 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 5 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 3 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 3 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 3 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 3 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256

# MNIST
mpirun -np 9 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 9 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 9 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 9 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 5 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 5 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 5 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 5 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 3 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 3 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -e X -s ? -m 2 -b 128
mpirun -np 3 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 3 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -ge X -s ? -m 4 -le 8 -b 256

# UNSW
mpirun -np 9 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 9 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -e X -s ? -m 2 -b 128
mpirun -np 9 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 9 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 5 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 5 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -e X -s ? -m 2 -b 128
mpirun -np 5 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 5 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -ge X -s ? -m 4 -le 8 -b 256
mpirun -np 3 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -e X -s ? -m 1 -b 2048 -m 1 -b 2048
mpirun -np 3 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -e X -s ? -m 2 -b 128
mpirun -np 3 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -ge X -s ? -m 3 -le 6 -a 0.2 -b 256
mpirun -np 3 python3 federated_learning.py -d UNSW -o Adam -lr 0.01 -ge X -s ? -m 4 -le 8 -b 256


