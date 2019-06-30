import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import websocket
import ssl


################ Step 1 Load Resnet Model
print("Step 1 -- Loading Resnet Model")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting : 
        for param in model.parameters() : 
            param.requires_grad = False

## Dataload            
image_size = 224
data_transforms = {
    'val': transforms.Compose([
    	transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

## Model Inicialize
model_ft = models.resnet152(pretrained = True)     
set_parameter_requires_grad(model_ft, False)
num_ftrs = model_ft.fc.in_features
num_classes = 25
model_ft.fc = nn.Linear(num_ftrs, num_classes)

# Load Weights
model_ft.load_state_dict(torch.load('./netFL.pt'))
model_ft = model_ft.to(device)

print("Step 1 -- Resnet Model Loaded")


################ Step 2 Open WebSocket Connection
print("Step 2 -- Opening WebSocket Connection")

ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
ws.connect("wss://tamats.com:55000/lgraph2")

print("Step 2 -- WebSocket Connection Opened")


##############################################################################


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 120)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cv2.namedWindow("test")

cascPath = "./haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

while True:
	ret, frame = cam.read()

	########### Detection ###########
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	################################

	if not ret:
		break

	k = cv2.waitKey(1)

	# ESC pressed
	if k%256 == 27:
		print("Escape hit, closing...")
		break
	
	
	if(len(faces)>0):

		################ Step 0 Cut Face from Frame
		transform = data_transforms['val']
		x,y,w,h = faces[0]
		#Cut using detection coordinates and change bgr to rgb
		frameRGB = cv2.cvtColor(frame[y-10:y+h+10, x-10:x+w+10], cv2.COLOR_BGR2RGB)
		#Apply transforms 
		frameToTensor = transform(frameRGB)
		#Add the 4th dimension
		frameToTensor.unsqueeze_(0)

		################ Step 1 Predict Coordinates
		outputs3D = []
		inputs = frameToTensor.to(device)
		model_ft.eval()
		with torch.set_grad_enabled(False) : 
			outputs = model_ft(inputs)

	    
		out = outputs.detach().to('cpu')
		out = out.numpy()
		for i in range(out.shape[0]) : 
			output3D = out[i]
			outputs3D.append(output3D)


		################ Step 2 Send Coordinates   
		for x in range(len(outputs3D)):

			weight = np.float32(outputs3D)
			arrayText = ",".join(map(str,weight[0]*1000))
			arrayText = "["+arrayText+"]"

			#Send to WebGlStudio
			ws.send('{"type":0, "channel":1, "data":'+arrayText+'}')


		################ Step 3 Show Rectangle 
		x,y,w,h = faces[0]
		cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)

	cv2.imshow("Test", frame)		
	
cam.release()
cv2.destroyAllWindows()
ws.close()

