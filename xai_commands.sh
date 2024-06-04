#python xai_tabular.py -d IOT_DNL -x PI
#python xai_tabular.py -d IOT_DNL -x PDV
#python xai_tabular.py -d Slicing5G -x PI
#python xai_tabular.py -d Slicing5G -x PDV
#python xai_tabular.py -d UNSW -x PI
#python xai_tabular.py -d UNSW -x PDV

#python xai_images.py -d CIFAR -x gradCAM
#python xai_images.py -d CIFAR -x integratedGrads
#python xai_images.py -d MNIST -x gradCAM
#python xai_images.py -d MNIST -x integratedGrads
#python xai_images.py -d FASHION -x gradCAM
#python xai_images.py -d FASHION -x integratedGrads

#python correlation_tabular.py -d IOT_DNL -x PI
#python correlation_tabular.py -d IOT_DNL -x PDV
#python correlation_tabular.py -d Slicing5G -x PI
#python correlation_tabular.py -d Slicing5G -x PDV
#python correlation_tabular.py -d UNSW -x PI
#python correlation_tabular.py -d UNSW -x PDV

python correlation_images.py -d CIFAR -x gradCAM
#python correlation_images.py -d CIFAR -x integratedGrads
python correlation_images.py -d MNIST -x gradCAM
#python correlation_images.py -d MNIST -x integratedGrads
python correlation_images.py -d FASHION -x gradCAM
#python correlation_images.py -d FASHION -x integratedGrads
