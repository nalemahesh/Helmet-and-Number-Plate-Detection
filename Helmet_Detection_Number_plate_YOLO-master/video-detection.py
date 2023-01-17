import cv2
import numpy as np
import requests 

net = cv2.dnn.readNet('yolov3-obj_2400.weights','yolov3-obj.cfg')
classes = []

with open('obj.names','r') as f:
    classes = f.read().splitlines()

# Now for detecting from Video (mp4)
cap = cv2.VideoCapture(0)

# Now for detecting from Video (mp4)
#cap =  cv2.VideoCapture(0, cv2.CAP_DSHOW)

msg="helmet not detected detected..So you have fine 500Rs."
def sms_send():
    url="https://www.fast2sms.com/dev/bulk"
    params={
 
        "authorization":"9AyekVcIf5m1r6zMnQlxvSgh0ZLs4PUJFERDXbCaq8NjOGdHW7MkSoYNij6xF7sp5TzWVl3Ldf2A9Zty",
        "sender_id":"SMSINI",
        "message":msg,
        "language":"english",
        "route":"p",
        "numbers":""
    }
    rs=requests.get(url,params=params)    
def numberplate():
    import pytesseract
    import cv2 
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    front_scale = 1.5
    font= cv2.FONT_HERSHEY_PLAIN
    cap= cv2.VideoCapture(0)
    cntr =0;
    while True:
        ret, frame =cap.read()
        cntr= cntr+1;
        if ((cntr%20)==0):
            
                imgH, imgW,_ = frame.shape
                
                x1,y1,w1,h1 = 0,0,imgH,imgW
                
                imgchar =pytesseract.image_to_string(frame)
                print(imgchar)
                imgboxes= pytesseract.image_to_boxes(frame)
                for boxes in imgboxes.splitlines():
                    boxes= boxes.split(' ')
                    x,y,w,h= int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
                    cv2.rectangle(frame, (x, imgH-y), (w, imgH-h),(0,0,255),3)
                    font= cv2.FONT_HERSHEY_SIMPLEX
                    cv2.imshow('Text detection',frame)
                    cv2.putText(frame, imgchar, (x1 + int(w1/80),y1 + int(h1/80)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
                    #no_plate = imgchar.replace(" ", "")
                   # print(no_plate)
                    sms_send() 
                    
                        
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
while True:
    _,img = cap.read()
    # print(classes)
    height, width, _ = img.shape
    # With this part we can open image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutPuts = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutPuts:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Print how many object is detected
    print(len(boxes))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes.flatten())
    # Now we need to show more information in a picture
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    # Loop for all object detected
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
    else:
        cap.release()
        numberplate()
        
        
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
