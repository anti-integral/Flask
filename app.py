import os
import numpy as np
from flask import Flask, render_template, request
import functions
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__, static_folder='static')

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST': 
        img_file = request.files['image']
        img_path = "static/" + img_file.filename    
        img_file.save(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (416, 416))
        label, bbox, confidence = functions.yolo(img_path)
        for i, label in enumerate(label):
            cv2.rectangle(img, (int(bbox[i][0]), int(bbox[i][1])), 
                          (int(bbox[i][2]), int(bbox[i][3])), (0, 255, 0), 2)
            cv2.putText(img, label + " (" + str(confidence[i]) + ")", 
                        (int(bbox[i][0]),int(bbox[i][1])-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        output_path = "static/output_" + img_file.filename
        output_name = "output_" + img_file.filename
        cv2.imwrite(output_path, img)
        
        return render_template("index.html", img_name=img_file.filename, output_name=output_name)

if __name__=='__main__':
    app.run(debug=True, host='localhost', port=5000)
