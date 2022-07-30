import streamlit as st
import cv2
from PIL import Image
import requests
import json

def get_color(label):
    """Helper method"""
    if label == "mask":
        return (0, 255, 0)
    return (0, 0, 255)

## setup slider for video upload
url = st.sidebar.text_input('NVIR Endpoint URL', 'http://192.168.0.123:52010/v1/infer/111111111')

st.sidebar.write('#### Select a video to upload.')
uploaded_video = st.sidebar.file_uploader("Choose video", type=["mp4"])
st.sidebar.write('[Nilvana vision inference runtime](https://nilvana.tw/products/nilvana-vision-inference-runtime)')

image = Image.open('./assets/nilvana_logo.png')
st.sidebar.image(image, width=256)

## setup main window
st.title('Inference Runtime Demo')

stframe = st.empty()
placeholder = st.empty()
if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    # save video to disk
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read())

    # load video from disk
    vidcap = cv2.VideoCapture(vid)

    while True:
        ret, frame = vidcap.read()
        if ret:
            frame = cv2.resize(frame, (1024, 540)) # change me!
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # inference
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            response = requests.post(url, files=files)
            result = response.json()

            # draw the prediction on the frame
            for obj in result:
                cv2.rectangle(frame, (int(obj['xmin']), int(obj['ymin'])), (int(obj['xmin']+obj['width']), int(obj['ymin']+obj['height'])), get_color(obj['label']), 2)
                label = "{}: {:.2f}%".format(obj['label'], obj['confidence'] * 100)
                frame = cv2.putText(frame, label, (int(obj['xmin']), int(obj['ymin']-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, get_color(obj['label']), 2)

            # display
            stframe.image(frame)
            with placeholder.container():
                # Display the JSON in main window.
                st.write('### JSON Output')
                st.write(result)
        else:
            break