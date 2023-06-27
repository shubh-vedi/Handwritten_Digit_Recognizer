import gradio as gr
from tensorflow import keras as k
import numpy as np

loaded_CNN = k.models.load_model('Digit_classification_model2.h5') 

def predict(img):
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28)
    img_array = img_array/255
    pred = loaded_CNN.predict(img_array)
    print(pred)
    return np.argmax(pred)

iface = gr.Interface(predict, inputs = 'sketchpad',
                     outputs = 'text',
                     allow_flagging = 'never',
                     description = 'Project : Recognizing hardwritten digits : Draw a Single Digit Below... (Draw in the middle for Better results)')
                    
iface.launch(debug = "True", width = 500, height = 500)



