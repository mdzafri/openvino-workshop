import cv2, datetime, sys # import opencv
import tensorflow.keras as keras
import numpy as np

if(len(sys.argv) != 4):
    print("teachable_img_openvino_classify.py /path/frozen_model.xml /path/frozen_model.bin /path/to/labels.txt")

np.set_printoptions(suppress=True)

#Load the saved model
modelPath = sys.argv[1]
webcam = cv2.VideoCapture(0)
model = keras.models.load_model(modelPath)
data_for_model = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def load_labels(path):
	f = open(path, 'r')
	lines = f.readlines()
	labels = []
	for line in lines:
		labels.append(line.split(' ')[1].strip('\n'))
	return labels

label_path = 'labels.txt'
labels = load_labels(label_path)
print(labels)

# This function proportionally resizes the image from your webcam to 224 pixels high
def image_resize(image, height, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# this function crops to the center of the resize image
def cropTo(img):
    size = 224
    height, width = img.shape[:2]

    sideCrop = (width - 224) // 2
    return img[:,sideCrop:(width - sideCrop)]

while True:
    ret, frame = webcam.read()
    if ret:
        #same as the cropping process in TM2
        img = image_resize(frame, height=224)
        img = cropTo(img)

        # flips the image
        img = cv2.flip(img, 1)

        #normalize the image and load it into an array that is the right format for keras
        normalized_img = (img.astype(np.float32) / 127.0) - 1
        data_for_model[0] = normalized_img
 
        #run inference
        # Start sync inference
        start_time = datetime.datetime.now()
        prediction = model.predict(data_for_model)
        end_time = datetime.datetime.now()
        time_st = 'Processing time: {:.2f} ms'.format(round((end_time - start_time).total_seconds() * 1000), 2)

        classid_str = "Top: "
        for i in range(0, len(prediction[0])):
            probs = np.squeeze(prediction[0])
            top_ind = np.argsort(probs)[-10:][::-1]
            for id in top_ind:
                classid_str += str(labels[id]) + ', ' + str(prediction[0][id])
                break
            break
            # print('{}: {}'.format(labels[i], prediction[0][i]))
            # classid_str = str(labels[i]) + ', ' + str(prediction[0][i])

        cv2.putText(frame,classid_str, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame,time_st, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()