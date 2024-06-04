import tensorflow as tf
import numpy as np

# Adapted from: 
# https://keras.io/examples/vision/integrated_gradients/
# https://www.tensorflow.org/tutorials/interpretability/integrated_gradients


class IntegratedGrads:

    def __init__(self, model, img_size):
        self.model = model
        self.img_size = img_size


    def get_gradients(self, img_input, top_pred_idx):
        img_input = tf.cast(img_input, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_input)
            preds = self.model(img_input)
            top_class = preds[:, top_pred_idx]

        grads = tape.gradient(top_class, img_input)
        return grads


    def gradients(self, img_input, baseline=None, num_steps=50):
        img_input = np.expand_dims(img_input, axis=0)
        top_pred_idx = np.argmax(self.model.predict(img_input, verbose=0))

        # Generate baseline image (all back)
        baseline = np.zeros(self.img_size).astype(np.float32)

        # Interpolation between baseline and input
        img_input = img_input.astype(np.float32)
        interpolated_image = [
            baseline + (step / num_steps) * (img_input - baseline)
            for step in range(num_steps + 1)
        ]
        interpolated_image = np.array(interpolated_image).astype(np.float32)

        # Calculate gradients for each interpolated image
        grads = []
        for img in interpolated_image:
            grad = self.get_gradients(img, top_pred_idx)
            grads.append(grad[0])
        grads = tf.convert_to_tensor(grads, dtype=tf.float32)

        # Approximate the integral using the trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)

        # Calculate integrated gradients
        integrated_grads = (img_input - baseline) * avg_grads
        return integrated_grads.numpy()[0]
    

    def heatmap(self, img_input, baseline=None, num_steps=50):
        grads = self.gradients(img_input, baseline, num_steps)

        if len(self.img_size) == 2:
            heatmap = np.abs(grads)
        else:
            heatmap = np.sum(np.abs(grads), axis=-1)

        heatmap /= np.max(heatmap)
        return heatmap
