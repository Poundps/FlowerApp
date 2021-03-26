import tensorflow as tf
from tensorflow.keras.models import load_model
import uuid
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS
import numpy as np
import io
from PIL import Image
import json
import base64
import redis
import os

app = Flask(__name__)

INPUT_SIZE = 299
CORS(app) 
model = load_model('flowers.h5')
print('!!! model loaded')

@app.route("/prediction", methods=['GET','POST'])
def flowers():
    r = redis.Redis(host='redis-flowers-api', port=6379, db=0)
    if request.method == 'GET':

        respone = []
        flower=[]
        keysFlower = r.keys('name:*')
        for keyF in keysFlower:
            flower.append(str(r.mget(keyF)))

        keysFilename = r.keys('image:*')
        counter = 0
        for key in keysFilename:
            #r.delete(key)
            
            flowersObj = r.mget(key)
            respone.append(
                {str(counter): {"filename": str(key), "flower": str(flower[counter]), "flowerImg": str(flowersObj)}})
            counter += 1

        return jsonify(respone)


    
    image = request.files["file"].read()
    im = Image.open(io.BytesIO(image))
    im = im.resize((INPUT_SIZE,INPUT_SIZE))

    filename = str(uuid.uuid4())
    im.save("../data/"+filename+".jpg")
    
    if len(im.size) == 2:
        im = im.convert("RGB")

    im_arr = np.array(im)
    im_arr_scaled = im_arr / 255.0 #3D

    im_arr_scaled_expand = np.expand_dims(im_arr_scaled, axis=0) #4D

    print('!!! preprocess complete')
    probs = model.predict(im_arr_scaled_expand)
    print('!!! inference complete')
    flowers = {
        0: 'daisy',
        1: "dandelion",
        2: 'rose',
        3: 'sunflower',
        4: 'tulip'
    }
    response = {
        'filename': filename,
        'prediction': {
            'daisy' : float(probs[0][0]),
            'dandelion' : float(probs[0][1]),
            'rose' : float(probs[0][2]),
            'sunflower' : float(probs[0][3]),
            'tulip' : float(probs[0][4]),
            'flowers': flowers[np.argmax(probs[0,:])]
        }
    }
    #save the prediction of the flower name in redis
    r.set("name:"+filename, flowers[np.argmax(probs[0, :])])
    r.set("key:"+filename, filename)
    r.set("image:"+filename, base64.encodebytes(image).decode('utf-8'))

    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
