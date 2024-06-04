#python3 single_training.py -d Slicing5G -o Adam -e 10 -lr 0.001 -b 1024
#python3 single_training.py -d IOT_DNL -o Adam -e 300 -lr 0.001 -b 1024
#python3 single_training.py -d CIFAR -o Adam -e 100 -lr 0.001 -b 32
#python3 single_training.py -d FASHION -o Adam -e 50 -lr 0.001 -b 32
#python3 single_training.py -d MNIST -o Adam -e 10 -lr 0.001 -b 32
#python3 single_training.py -d UNSW -o Adam -e 30 -lr 0.001 -b 1024

## CIFAR
#mpirun -np 9 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -e 300 -s 0.78 -m 1 -b 128
#mpirun -np 9 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -e 200 -s 0.78 -m 2 -b 32
#mpirun -np 9 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -ge 500 -s 0.78 -m 3 -le 1 -a 0.3 -b 32
#mpirun -np 9 python3 federated_learning.py -d CIFAR -o Adam -lr 0.001 -ge 200 -s 0.78 -m 4 -le 1 -b 32

# FASHION
#mpirun -np 9 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -e 100 -s 0.91 -m 1 -b 64
#mpirun -np 9 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -e 100 -s 0.91 -m 2 -b 32
#mpirun -np 9 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -ge 100 -s 0.91 -m 3 -le 1 -a 0.2 -b 32
#mpirun -np 9 python3 federated_learning.py -d FASHION -o Adam -lr 0.001 -ge 100 -s 0.91 -m 4 -le 1 -b 32


## MNIST
mpirun -np 9 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -e 30 -s 0.98 -m 1 -b 128
#mpirun -np 9 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -e 30 -s 0.98 -m 2 -b 32
#mpirun -np 9 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -ge 50 -s 0.98 -m 3 -le 1 -a 0.2 -b 32
#mpirun -np 9 python3 federated_learning.py -d MNIST -o Adam -lr 0.001 -ge 50 -s 0.98 -m 4 -le 1 -b 32


## UNSW
#mpirun -np 9 python3 federated_learning.py -d UNSW -o Adam -lr 0.001 -e 30 -s 0.95 -m 1 -b 64
#mpirun -np 9 python3 federated_learning.py -d UNSW -o Adam -lr 0.001 -e 30 -s 0.95 -m 2 -b 32
#mpirun -np 9 python3 federated_learning.py -d UNSW -o Adam -lr 0.001 -ge 30 -s 0.95 -m 3 -le 1 -a 0.2 -b 32
#mpirun -np 9 python3 federated_learning.py -d UNSW -o Adam -lr 0.001 -ge 30 -s 0.95 -m 4 -le 1 -b 32


# Slicing5G
#mpirun -np 9 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -e 30 -s 1 -m 1 -b 64
#mpirun -np 9 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -e 30 -s 1 -m 2 -b 32
#mpirun -np 9 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -ge 30 -s 1 -m 3 -le 1 -a 0.2 -b 32
#mpirun -np 9 python3 federated_learning.py -d Slicing5G -o Adam -lr 0.001 -ge 30 -s 1 -m 4 -le 1 -b 32


# IOT_DNL 
#mpirun -np 9 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -e 30 -s 0.99 -m 1 -b 64
#mpirun -np 9 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -e 30 -s 0.99 -m 2 -b 32
#mpirun -np 9 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -ge 30 -s 0.99 -m 3 -le 1 -a 0.2 -b 32
#mpirun -np 9 python3 federated_learning.py -d IOT_DNL -o Adam -lr 0.001 -ge 30 -s 0.99 -m 4 -le 1 -b 32
