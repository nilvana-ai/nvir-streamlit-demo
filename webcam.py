import streamlit as st
import cv2
from PIL import Image
import requests
import json

def run(url:str):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (854, 480)) # change me!
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # inference
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            response = requests.post(url, files=files)
            result = response.json()

            # draw the prediction on the frame
            for obj in result:
                if obj['confidence'] * 100 > threshold:
                    cv2.rectangle(frame, (int(obj['xmin']), int(obj['ymin'])), (int(obj['xmin']+obj['width']), int(obj['ymin']+obj['height'])), (0, 255, 0), 2)
                    label = "{}: {:.2f}%".format(obj['label'], obj['confidence'] * 100)
                    frame = cv2.putText(frame, label, (int(obj['xmin']), int(obj['ymin']-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # display
            stframe.image(frame)
        else:
            break

if __name__ == '__main__':
    # nvir url
    url = st.sidebar.text_input('NVIR Endpoint URL', 'http://127.0.0.1:52010/v1/infer/111111111')
    
    # threshold
    threshold = st.sidebar.slider('Threshold', 0, 100, 60)
    
    # click to start
    if st.sidebar.button('Detect'):
        run(url)
    
    # hyperlink to nvir website
    st.sidebar.write('[Nilvana™ vision inference runtime](https://nilvana.tw/products/nilvana-vision-inference-runtime)')

    # logo
    image = Image.open('./assets/nilvana_logo.png')
    st.sidebar.image(image, width=256)

    st.title('Nilvana™ Vision Inference Runtime')    

