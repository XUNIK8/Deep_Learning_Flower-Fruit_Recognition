import numpy as np
import tensorflow as tf
from PIL import Image
def imgToMat_RGB(img):            
    img = Image.open(img)
    img = img.convert("RGB")
    #data = img.getdata()
    img = img.resize((192,192))
    mat = np.array(img)
    mat=mat/255.0
    mat=2*mat-1
    mat = mat.astype(np.float32)
    return mat


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
while(True):
    i+=1
    os.system("echo 'Take %sth photo...'" % (i))
    os.system("raspistill -t 1000 -o img.jpg")
    os.system("echo 'Recognition...'")
	img = imgToMat_RGB("img.jpg")
	# Test model on random input data.
	input_shape = input_details[0]['shape']
	l = []
	l.append(img)
	l = np.array(l)
	#print(input_shape)
	#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float64)
	#print(input_data)
	interpreter.set_tensor(input_details[0]['index'], l)

	interpreter.invoke()
	# The function `get_tensor()` returns a copy of the tensor data.
	# Use `tensor()` in order to get a pointer to the tensor.
	output_data = interpreter.get_tensor(output_details[0]['index'])
	print(output_data)
    bk_img = cv2.imread("img.jpg")
    cv2.putText(bk_img,class_name,(200,200),cv2.FONT_HERSHEY_SIMPLEX,7,(0,0,0),15,cv2.LINE_AA)
    cv2.namedWindow("image",0)
    cv2.resizeWindow("image",800,600)
    cv2.imshow("image",bk_img)
    cv2.waitKey()
    cv2.imwrite("img.jpg",bk_img)
