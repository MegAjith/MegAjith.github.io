import tensorflow as tf
import numpy as np
import os 
import keras
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from keras.preprocessing.image import ImageDataGenerator
import json
import glob
import numpy as np
import shutil
from keras.models import model_from_json


with open('/Users/megha/Projects/model_architecture.json', 'r') as json_file:
    model = model_from_json(json_file.read())
model.load_weights('model_weights.h5')
model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])


app = Flask(__name__)
  
app.config['UPLOAD_FOLDER'] = 'C:/Users/megha/Projects/uploaded/image/'
  
@app.route('/')
def upload_f():
    return render_template('upload.html')
  
def finds():
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_dir = 'uploaded'
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size =(250, 400),
            color_mode ="rgb",
            shuffle = False,
            class_mode ='binary',
            batch_size = 1)
    pred = model.predict(test_generator)
    print(pred)
    return pred
    #return str(vals[np.argmax(pred)])
  
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        cwd=os.path.join(os.getcwd(),"uploaded","image")
        f.save(os.path.join(cwd,secure_filename(f.filename)))
        try:
            val = finds()
        except Exception as e:
            val=e
        shutil.rmtree('C:/Users/megha/Projects/uploaded/image/')
        os.mkdir('C:/Users/megha/Projects/uploaded/image/')
        return render_template('pred.html', ss =  "%.2f" % (val.item()*100))

if __name__ == '__main__':
    app.run()
