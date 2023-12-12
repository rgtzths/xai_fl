mpirun -np 9 -hostfile FL/host_file python3 federated_learning.py -m 1 -d IOT_DNL -lr 0.000005
mpirun -np 9 -hostfile FL/host_file python3 federated_learning.py -m 2 -d IOT_DNL -lr 0.000005
mpirun -np 9 -hostfile FL/host_file python3 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005
mpirun -np 9 -hostfile FL/host_file python3 federated_learning.py -m 4 -d IOT_DNL -lr 0.000005
mpirun -np 9 -hostfile FL/host_file python3 federated_learning.py -m 1 -d Slicing5G -lr 0.00001
mpirun -np 9 -hostfile FL/host_file python3 federated_learning.py -m 2 -d Slicing5G -lr 0.00001
mpirun -np 9 -hostfile FL/host_file python3 federated_learning.py -m 3 -d Slicing5G -lr 0.00001
mpirun -np 9 -hostfile FL/host_file python3 federated_learning.py -m 4 -d Slicing5G -lr 0.00001
python3 single_training.py -d IOT_DNL -lr 0.000005
python3 single_training.py -d Slicing5G -lr 0.00001
mpirun -np 5 -hostfile FL/host_file python3 federated_learning.py -m 1 -d IOT_DNL -lr 0.000005
mpirun -np 5 -hostfile FL/host_file python3 federated_learning.py -m 2 -d IOT_DNL -lr 0.000005
mpirun -np 5 -hostfile FL/host_file python3 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005
mpirun -np 5 -hostfile FL/host_file python3 federated_learning.py -m 4 -d IOT_DNL -lr 0.000005
mpirun -np 5 -hostfile FL/host_file python3 federated_learning.py -m 1 -d Slicing5G -lr 0.00001
mpirun -np 5 -hostfile FL/host_file python3 federated_learning.py -m 2 -d Slicing5G -lr 0.00001
mpirun -np 5 -hostfile FL/host_file python3 federated_learning.py -m 3 -d Slicing5G -lr 0.00001
mpirun -np 5 -hostfile FL/host_file python3 federated_learning.py -m 4 -d Slicing5G -lr 0.00001
mpirun -np 3 -hostfile FL/host_file python3 federated_learning.py -m 1 -d IOT_DNL -lr 0.000005
mpirun -np 3 -hostfile FL/host_file python3 federated_learning.py -m 2 -d IOT_DNL -lr 0.000005
mpirun -np 3 -hostfile FL/host_file python3 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005
mpirun -np 3 -hostfile FL/host_file python3 federated_learning.py -m 4 -d IOT_DNL -lr 0.000005
mpirun -np 3 -hostfile FL/host_file python3 federated_learning.py -m 1 -d Slicing5G -lr 0.00001
mpirun -np 3 -hostfile FL/host_file python3 federated_learning.py -m 2 -d Slicing5G -lr 0.00001
mpirun -np 3 -hostfile FL/host_file python3 federated_learning.py -m 3 -d Slicing5G -lr 0.00001
mpirun -np 3 -hostfile FL/host_file python3 federated_learning.py -m 4 -d Slicing5G -lr 0.00001