#!/usr/bin/env python
from openvino.inference_engine import IECore
import numpy as np
import cv2, datetime, sys

if(len(sys.argv) != 4):
    print("teachable_img_openvino_classify.py /path/frozen_model.xml /path/frozen_model.bin /path/to/labels.txt")

# OpenVINO Plugin Initialization
ie = IECore()

# Read IR
net = ie.read_network(model=sys.argv[1], weights=sys.argv[2])

# Prepare blobs
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

# Read and pre-process input images
n, c, h, w = net.inputs[input_blob].shape

# Loading model to the plugin
exec_net = ie.load_network(network=net, device_name="CPU")

# Process output blob and pretty display
with open(sys.argv[3], 'r') as f:
    labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frame.shape[:-1] != (h, w):
        image = cv2.resize(frame, (w,h))

    image = (np.asarray(image).astype(np.float32)/127.0) - 1 # convert to numpy arrray and normalize the image
    image = image.transpose((2, 0, 1)) # change layout from HWC to CHW

    # Start sync inference
    start_time = datetime.datetime.now()
    res = exec_net.infer(inputs={input_blob: image})
    end_time = datetime.datetime.now()

    # Process output blob and pretty display
    res = res[out_blob]
    classid_str = "Top: "
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-10:][::-1]
        for id in top_ind:
            classid_str += labels_map[id]
            break
        break

    time_st = 'Processing time: {:.2f} ms'.format(round((end_time - start_time).total_seconds() * 1000), 2)

    cv2.putText(frame,classid_str, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame,time_st, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
