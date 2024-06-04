# source ../nn_analysis/venv/bin/activate

python3 data_processing.py -d IOT_DNL
python3 data_processing.py -d Slicing5G
python3 data_processing.py -d MNIST
python3 data_processing.py -d CIFAR
python3 data_processing.py -d FASHION
python3 data_processing.py -d UNSW

python3 data_division.py -d IOT_DNL -n 8
python3 data_division.py -d Slicing5G -n 8
python3 data_division.py -d MNIST -n 8
python3 data_division.py -d CIFAR -n 8
python3 data_division.py -d FASHION -n 8
python3 data_division.py -d UNSW -n 8

python3 data_division.py -d IOT_DNL -n 4
python3 data_division.py -d Slicing5G -n 4
python3 data_division.py -d MNIST -n 4
python3 data_division.py -d CIFAR -n 4
python3 data_division.py -d FASHION -n 4
python3 data_division.py -d UNSW -n 4

python3 data_division.py -d IOT_DNL -n 2
python3 data_division.py -d Slicing5G -n 2
python3 data_division.py -d MNIST -n 2
python3 data_division.py -d CIFAR -n 2
python3 data_division.py -d FASHION -n 2
python3 data_division.py -d UNSW -n 2