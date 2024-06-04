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
            category_id = np.argmax(predictions[0])
            output = predictions[:, category_id]
            grads = tape.gradient(output, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat

        return np.squeeze(heatmap)
