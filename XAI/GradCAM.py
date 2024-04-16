import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Adapted from: 
# https://github.com/samson6460/tf_keras_gradcamplusplus
# https://keras.io/examples/vision/grad_cam/


class GradCAM:
    
    def __init__(self, model, img_size):
        self.model = model
        self.img_size = img_size
        self.last_conv_layer = self.get_last_conv_layer()
        self.grad_model = self.get_grad_model()


    def get_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if layer.name.startswith('conv'):
                return layer
        return None


    def get_grad_model(self):
        return tf.keras.models.Model(self.model.inputs, [self.last_conv_layer.output, self.model.output])
    

    def heatmap(self, img):
        img = np.expand_dims(img, axis=0)

        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(img)
            pred_index = np.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        # heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        return heatmap
