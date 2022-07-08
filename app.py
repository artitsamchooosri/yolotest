"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
from IPython.display import display
import cv2
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)


#'''
# Load Pre-trained Model
#model = torch.hub.load(
       # "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
      #  )#.autoshape()  # force_reload = recache latest code
#'''
# Load Custom Model
#model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.pt", force_reload=True).autoshape()
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
# Set Model Settings
model.eval()
model.conf = 0.45  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 
from io import BytesIO

def gen():
    cap=cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            #print(results)
            #print(results.pandas().xyxy[0])
            #results.pandas().xyxy data frame รายการที่ตรวจจับได้พร้อมระยะ

            #print(*results.pandas().xyxy, sep = "\n")
            #results.render()  # updates results.imgs with boxes and labels
            #results.print()  # print results to screen
            #results.show() 
            #print(results.imgs)
            #print(type(img))
            #print(results)
            #plt.imshow(np.squeeze(results.render()))
            #print(type(img))
            #print(img.mode)
            
            #convert remove single-dimensional entries from the shape of an array
            # create dataframe
            #print(type(results.pandas().xyxy))
            
           # df = pd.DataFrame([t.__dict__ for t in results.pandas().xyxy[0] ])

            #print(type(results.pandas().xyxy[0]))
            df=results.pandas().xyxy[0]
            df = df.sort_values(['ymin', 'xmin'],ascending=[False, True])
            display(df)
            #print(df)
            # set Frame width and height (640,384)
            (img_W, img_H) = (640, 384)
            # set Y for traffic checking zone
            (Y1, Y2) = (214, 324)

            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

            #print(type(img))
            #print(img.shape)
            #frame = img
            #ret,buffer=cv2.imencode('.jpg',img)
            #frame=buffer.tobytes()
            #print(type(frame))
            #for img in results.imgs:
                #img = Image.fromarray(img)
            #ret,img=cv2.imencode('.jpg',img)
            #img=img.tobytes()

            #encode output image to bytes
            #img = cv2.imencode('.jpg', img)[1].tobytes()
            #print(type(img))
        else:
            break
        #print(cv2.imencode('.jpg', img)[1])

        #print(b)
        #frame = img_byte_arr

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

