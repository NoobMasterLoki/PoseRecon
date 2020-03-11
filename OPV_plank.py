#Python code for running posture detection and correction for the exercise of Plank
#import the required libraries for running the script
import cv2
import time
import numpy as np
import math

MODE = "MPI"

if MODE is "COCO":  #to load coco prototext files
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :    #to load mpii prototext files
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


inWidth = 368   #width of video taken by openpose
inHeight = 368  #height of video taken by openpose
threshold = 0.1


input_source = "plank.mp4"  #change name for required video to be analyzed
cap = cv2.VideoCapture(input_source)    #to load the video
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter('output_'+input_source,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
ang_back_hip_knee=[]
q=0
while cv2.waitKey(1) < 0:
    q+=1
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    model_features=points

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    x=0
    for i in model_features:
        x+=1
        if i==None:
            if x!=1:
                model_features[x-1]=model_features[x-2]
            else:
                model_features[0]=(0,0)
                
                
                
    model_neck=model_features[1]
    model_right_hip=model_features[8]
    model_right_knee=model_features[9]
    model_right_ankle=model_features[10]
    model_left_hip=model_features[11]
    model_left_knee=model_features[12]
    model_left_ankle=model_features[13]
    model_back=model_features[14]
    
    a1_m=-(math.degrees(math.atan2(model_neck[1]-model_back[1], model_neck[0]-model_back[0])))
    a2_m=-(math.degrees(math.atan2(model_back[1]-model_right_hip[1], model_back[0]-model_right_hip[0])))
    a3_m=-(math.degrees(math.atan2(model_right_hip[1]-model_right_knee[1], model_right_hip[0]-model_right_knee[0])))
    a4_m=-(math.degrees(math.atan2(model_right_knee[1]-model_right_ankle[1], model_right_knee[0]-model_right_ankle[0])))
    a5_m=-(math.degrees(math.atan2(model_back[1]-model_left_hip[1], model_back[0]-model_left_hip[0])))
    a6_m=-(math.degrees(math.atan2(model_left_hip[1]-model_left_knee[1], model_left_hip[0]-model_left_knee[0])))
    a7_m=-(math.degrees(math.atan2(model_left_knee[1]-model_left_ankle[1], model_left_knee[0]-model_left_ankle[0])))
    #the angles in the posture for correction are calculated here 
    a12_m=abs(a1_m-a2_m)#back
    a23_m=abs(a2_m-a3_m)#hip
    a34_m=abs(a3_m-a4_m)#knee
    a15_m=abs(a1_m-a5_m)#back
    a56_m=abs(a5_m-a6_m)#hip
    a67_m=abs(a6_m-a7_m)#knee
    #formatting the data for comparison and correction
    if(a12_m>90):
        while(a12_m>90):
            a12_m=abs(a12_m-180)
    if(a15_m>90):
        while(a15_m>90):
            a15_m=abs(a15_m-180)
    if(a23_m>90):
        while(a23_m>90):
            a23_m=abs(a23_m-180)
    if(a56_m>90):
        while(a56_m>90):
            a56_m=abs(a56_m-180)
    if(a34_m>90):
        while(a34_m>90):
            a34_m=abs(a34_m-180)
    if(a67_m>90):
        while(a67_m>90):
            a67_m=abs(a67_m-180)
    
    a_m=[a12_m,a23_m,a34_m,a15_m,a56_m,a67_m]
    ang_back_hip_knee.append(a_m)
    #print(a1_m,a2_m,a3_m,a4_m,a5_m,a6_m,a7_m)
    print(a_m)
    
    
    if((a34_m<=10 or a67_m<=10) and (a12_m<=10 or a15_m<=10) and (a23_m<=10 or a56_m<=10)):
        print("Good Plank")#angle of the back checked at every frame, and required correction steps provided
        if (a12_m<=a15_m):  #once a determined as good plank the hip and knee angles are checked
            print("Just",a12_m," degrees for a perfectly straight back")
        else:
            print("Just",a15_m," degrees for a perfectly straight back")
        if (a34_m<=a67_m):
            print("Knee bend=",a34_m)
        else:
            print("Knee bend=",a67_m)
        if (a23_m<=a56_m):
            print("Hip bend=",a23_m)
        else:
            print("Hip bend=",a56_m)
    else:
        print("Bad Plank")
        if (a12_m>10 and a15_m>10):
            print("Keep a straight back")
        if (a12_m<=a15_m):
            print("Just",a12_m," degrees for a perfectly straight back")
        else:
            print("Just",a15_m," degrees for a perfectly straight back")
        if (a34_m>10 and a67_m>10):
            print("Do not bend your knees")
        if (a34_m<=a67_m):
            print("Knee bend=",a34_m)
        else:
            print("Knee bend=",a67_m)
        if (a23_m>10 and a56_m>10):
            print("Maintain the hip in the line of a straight back and unbent knee")
        if (a23_m<=a56_m):
            print("Hip bend=",a23_m)
        else:
            print("Hip bend=",a56_m)


    

    #cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow('Output-Skeleton', frame)
    print('frame ',q)
    vid_writer.write(frame)
print(ang_back_hip_knee)

vid_writer.release()
