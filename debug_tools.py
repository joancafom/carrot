import cv2
import keras.backend as K
from keras.models import Model
import time
import numpy as np

def showImage(image, title = "Debug"):
    assert image is not None
    cv2.imshow(title, image)
    cv2.waitKey(1)

def showHeatmap(model, state, layer_name="conv4"):

    preds = model.predict(np.array([state]))
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("conv4")

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([np.array([state])])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    img = np.array(state)
    img = np.uint8(255 * img)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imshow("GradCam", superimposed_img)
    cv2.waitKey(1)

def showLayerOutput(model, state):

    layer_outputs = [layer.output for layer in model.layers[:4]][1:] 
    # Extracts the outputs of the top 4 layers

    # Creates a model that will return these outputs, given the model input
    activation_model = Model(inputs=model.input, outputs=layer_outputs) 
    activations = activation_model.predict(np.array([state])) 
    
    nth = 0
    nth_layer_activation = activations[nth]
    #print(first_layer_activation.shape)
    # Channel 1 of first activation layer
    image = nth_layer_activation[0, :, :, 1]
    showImage(image)
