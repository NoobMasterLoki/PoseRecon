#Python code for running posture detection and correction for the exercise of Squats
#import the required libraries for running the script
import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt

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


input_source = "squat.mp4"  #change name for required video to be analyzed
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
    a23_m=abs(180-abs(a2_m-a3_m))#hip
    a34_m=abs(180-abs(a3_m-a4_m))#knee
    a15_m=abs(a1_m-a5_m)#back
    a56_m=abs(180-abs(a5_m-a6_m))#hip
    a67_m=abs(180-abs(a6_m-a7_m))#knee
    #formatting the data for comparison and correction
    if(a12_m>90):
        while(a12_m>90):
            a12_m=abs(a12_m-180)
    if(a15_m>90):
        while(a15_m>90):
            a15_m=abs(a15_m-180)
    
    a_m=[a12_m,a23_m,a34_m,a15_m,a56_m,a67_m]
    ang_back_hip_knee.append(a_m)
    #print(a1_m,a2_m,a3_m,a4_m,a5_m,a6_m,a7_m)
    print(a_m)
    
    if (a12_m>10 and a15_m>10): #angle of the back checked at every frame, and required correction steps provided
        print("Keep a straight back")
        if (a12_m<=a15_m):
            print("Just",a12_m," degrees for a perfectly straight back")
        else:
            print("Just",a15_m," degrees for a perfectly straight back")
    
    
    

    #cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow('Output-Skeleton', frame)
    print('frame ',q)
    vid_writer.write(frame)
print(ang_back_hip_knee)
a=ang_back_hip_knee
h1=[x[1] for x in a]
#h2=[y[4] for y in a]
h1_mean=(min(h1)+max(h1))/2
#h2_mean=(min(h2)+max(h2))/2
#h1_h=[x for x in h1 if x<=h1_mean]
#h2_h=[y for y in h2 if y<=h2_mean]
m_h1=[h1_mean for i in range(1,len(a)+1)]
#m_h2=[h2_mean for i in range(1,len(a)+1)]
z=[r for r in range(1,len(a)+1)]
plt.plot(z, h1) 
plt.plot(z, m_h1)  
plt.xlabel('frames') 
plt.ylabel('Hip angle') 
# function to show the plot 
plt.show()
#c=0
c1=0
x=[]
for i in range(0,len(a)):#this loop is responsible for marking the lowest points of the hip angles in the video to be checked
    c1+=1
    if(h1[i]<=h1_mean):
        if(c1==1):
            x1=i
    if(h1[i]>h1_mean):
        if(c1>10):
            x2=i
            x.append(round((x1+x2)/2)+1)
            #print(x1,x2)
        c1=0 
print("Frames to check the squats=",x)
for i in x:#once the required lowest frames and determined the angle at these points is to be checked and provide correction
    [a12_m,a23_m,a34_m,a15_m,a56_m,a67_m]=a[i]
    print("\n",a[i])
        
    if((60<=a34_m<=80 or 60<=a67_m<=80) and (a12_m<=10 or a15_m<=10) and (40<=a23_m<=60 or 40<=a56_m<=60)):#the anlge of the hip is checked for beingin the required range to be classified as a good squat
        print("Good Squat")
        if (a12_m<=a15_m):#once determined that it is a good squat, other factors of the posture are checked like the back angles
            print("Just",a12_m," degrees for a perfectly straight back")
        else:
            print("Just",a15_m," degrees for a perfectly straight back")
        if (40<=a23_m<=60):
            print("Hip angle=",a23_m)
        elif (40<=a56_m<=60):
            print("Hip angle=",a56_m)
        if (60<=a34_m<=80):
            print("Knee angle=",a34_m)
        elif (60<=a67_m<=80):
            print("Knee angle=",a67_m)
    else:
        print("Bad Squat")#if determined as a bad deadlift, the factors contributing to that are mentioned and required corrections are provided
        if (a12_m>10 and a15_m>10):
            print("Keep a straight back")
        if (a12_m<=a15_m):
            print("Just",a12_m," degrees for a perfectly straight back")
        else:
            print("Just",a15_m," degrees for a perfectly straight back")
        if ((a23_m>60 or a23_m<40) and (a56_m>60 or a56_m<40)):
            print("Maintain the hip angle around 50 degrees")
            if (a23_m>60 and a56_m>60):
                print("Bend the torso a little more")
            if (a23_m<40 and a56_m<40):
                print("Bend the torso a little less")
        if (a23_m<=a56_m):
            print("Hip angle=",a23_m)
        else:
            print("Hip angle=",a56_m)
        
        if ((a34_m>80 or a34_m<60) and (a67_m>80 or a67_m<60)):
            print("Maintain the knee angle around 70 degrees")
            if (a34_m>80 and a67_m>80):
                print("Crouch a little more")
            if (a34_m<60 and a67_m<60):
                print("Crouch a little less")
        if (a34_m<=a67_m):
            print("Knee angle=",a34_m)
        else:
            print("Knee angle=",a67_m)


vid_writer.release()
