#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.patches as patches
from IPython.display import clear_output
import matplotlib.pylab as plt
import os

try:
    import google.colab
    IN_COLAB = True
    get_ipython().run_line_magic('matplotlib', 'inline')
    from google.colab import drive
    from google.colab.patches import cv2_imshow
    drive.mount('/content/gdrive')
    DRIVE_ROOT='/content/gdrive/My Drive/opencv/'
    get_ipython().system('pip install face_recognition')
except:
    IN_COLAB =False
    get_ipython().run_line_magic('matplotlib', 'notebook')
    DRIVE_ROOT='./'

import face_recognition
face_dir = os.path.join(DRIVE_ROOT,'known_face')
OUT_DIR =os.path.join(DRIVE_ROOT,'output')


# In[3]:


# detect face locations        
def detect_face_locations(image,do_gray=False,do_scale=False):
    if do_scale:
        small_frame = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    else:
        small_frame = image
    if do_gray:
        gray = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
        rgb_frame = gray[:,:,::-1]
    else:
        rgb_frame=small_frame[:,:,::-1]

#    face_locations = face_recognition.face_locations(rgb_frame,model="cnn")
    face_locations = face_recognition.face_locations(rgb_frame)
    for face_location in face_locations:
        (top,right,bottom,left)=face_location;
        if do_scale:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

        cv2.rectangle(image, (left,top), (right,bottom), (0,0,255),2)
        #plt.plot(face_location[1],face_location[0],'bo')
        #plt.plot(face_location[1],face_location[2],'bo')
        #plt.plot(face_location[3],face_location[2],'bo')
        #plt.plot(face_location[3],face_location[0],'bo')
        #plt.show()
    return image

def detect_face_save(image):
    rgb_frame=image[:,:,::-1]
    height,width = image.shape[:2]
    #print('width',width,'height',height)
#    face_locations = face_recognition.face_locations(rgb_frame,model="cnn")
    face_locations = face_recognition.face_locations(rgb_frame)
    n=len(face_locations)
    cols=5
    if n<=cols:
        cols=n
        rows=1
    else:
        cols=5
        rows=int(1+n/3)
    r=1
    c=1
    i=1
    #print(rows,cols)
    fig = plt.figure(figsize=(cols*3,rows*3))
    for face_location in face_locations:
        print(face_location)
        (top,right,bottom,left)=face_location;
        #print('left',left,'top',top,'right',right,'bottom',bottom)
        w = int((right - left)/2)
        h = int((bottom - top)/2) 
        print('w',w,'h',h)
        top=max(0,top-h-h)
        bottom = min(height-1,bottom+h)
        left=max(0,left-w)
        right=min(width-1,right+w)
        #print('left',left,'top',top,'right',right,'bottom',bottom)
        image_face=image[top : bottom, left : right]
        #cv2.rectangle(image, (left,top), (right,bottom), (255,0,0),2)
        img = cv2.cvtColor(image_face,cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(rows,cols,i)
        ax.imshow(img)
        c+=1
        if c> cols:
            c=0
            r+=1
#        plt.imshow(img)
#        plt.show()    
        i+=1
    plt.show()
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
image=cv2.imread('test-data/IMG_5710.JPG')
detect_face_save(image)
#image=detect_face_locations(image)
#cv2.imshow('show',image)
#if cv2.waitKey(0) & 0xFF:
#    cv2.destroyAllWindows()
img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


# In[12]:


known_face_encodings=[]
known_face_names=[]
subject_image_names=os.listdir(face_dir)
for image_name in subject_image_names:
    if image_name.startswith("."):
        continue
    name=os.path.splitext(image_name)[0];
    image_path = os.path.join(face_dir,image_name)
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    print(name)
    detect_face_locations(image)
    #img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
    #if IN_COLAB:
    #  cv2_imshow(image)
    #else:
    #  cv2.imshow('training',image)
    #  if cv2.waitKey(500) & 0xFF:
    #    cv2.destroyAllWindows()


# In[4]:


# single frame recognition
def recognize_face(frame,do_scale=False,threshold=0.5,debug=False):
    face_locations=[]
    face_encodings=[]
    face_names=[]
    
    if do_scale:
        # resize frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    else:
        small_frame = frame
    result=False   
    # convert the image from BGR color to RGB color
    rgb_small_frame = small_frame[:,:,::-1]
    
    # find all the faces and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    #if debug:
        #print(face_locations)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
    result=False
    for face_encoding in face_encodings:
        # see if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'Unknown'
        
        # use the known face with the smallest distance
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            if face_distance[best_match_index]<threshold:
                name = known_face_names[best_match_index]
            else:
                name = '?'+known_face_names[best_match_index]+'?'
            if debug:
                print(name,face_distance)
        if name == 'Toma':
            result=True
        face_names.append(name)
    
    #display result
    for(top,right,bottom,left),name in zip(face_locations,face_names):
        #scale back up face locations since the frame detected in was scaled to 1/4 size
        if do_scale:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
        
        #draw a box around the face
        cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255),2)
        
        # draw a label with a name below the face
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.2
        thickness = 2
        margin=1
        
        size = cv2.getTextSize(name, font, font_scale, thickness)
        text_width = size[0][0]
        text_height = size[0][1]
        
        cv2.rectangle(frame, (left, bottom), (left+text_width+margin, bottom+text_height+margin), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left,bottom + text_height), font,font_scale, (255,255,255), thickness)
    return result,frame


# In[5]:


# test functions
# format time(second) as HH:MM:ss.mmm
def format_time(sec):
    msec = (int)(sec * 1000.)
    s, ms = divmod(msec, 1000)
    m, s =divmod(s,60)
    return '{:02d}:{:02d}.{:03d}'.format(m,s,ms)

def format_time_forfile(sec):
    msec = (int)(sec * 1000.)
    s, ms = divmod(msec, 1000)
    m, s =divmod(s,60)
    return '{:02d}m{:02d}s{:03d}ms'.format(m,s,ms)

# single image
def single_image(filename,detect_only=False,debug=False):
#image=cv2.imread("img_interviewMovie.jpg")
    image=cv2.imread(filename)
    if detect_only:
        image=detect_face_locations(image)
    else:
        ret,image=recognize_face(image,debug=debug)

    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    
    file_name=os.path.join(OUT_DIR,os.path.basename(filename))
    cv2.imwrite(file_name,image)
    if IN_COLAB:
      cv2_imshow(image)
    else:
      cv2.imshow("sigle test",image)
      if cv2.waitKey(0) & 0xFF:
        cv2.destroyAllWindows()
    
def getInterval(fps):
    interval =1.0/fps
    time_wait = (int)(interval * 1000.0)
    print('interval: ',interval,'time_wait: ',time_wait)
    return interval,time_wait

def video_proc(filename,output=None,capture=False,detect_only=False,no_wait=False,view=True):
    if capture:
        pre = os.path.splitext(os.path.basename(filename))[0]
    cap = cv2.VideoCapture(filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval,time_wait = getInterval(fps)
    time_stamp=0.0
    frame_count=0
    if no_wait:
        time_wait=1
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    margin=1
    size = cv2.getTextSize(name, font, font_scale, thickness)
    text_width = size[0][0]
    text_height = size[0][1]
    counter = 0
    
    # VideoWriter を作成する。
    if output is not None:
        if IN_COLAB:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(os.path.join(OUT_DIR,output), fourcc, fps, (width, height))
    else:
        writer = None
        
    if capture:
      CAPTURE_DIR=os.path.join(OUT_DIR,pre)
      os.makedirs(CAPTURE_DIR,exist_ok=True)

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            cap.release()
            break
        frame_count += 1
        if detect_only:
            frame=detect_face_locations(frame)
        else:
            result,frame=recognize_face(frame)
        cv2.putText(frame,format_time(time_stamp),(margin,margin+text_height),font,font_scale,(255,0,0),thickness)
        if result:
            counter += 1
            print(str(counter),format_time(time_stamp))
            if capture:
                out_name=os.path.join(CAPTURE_DIR,pre+'_'+format_time_forfile(time_stamp)+'.png')
                cv2.imwrite(out_name,frame)
        if not IN_COLAB & view :
            cv2.imshow('video',frame)
            k = cv2.waitKey(time_wait) & 0xff
            if k == 27:
                break;
        if writer is not None:
            writer.write(frame)
        time_stamp += interval
    if not IN_COLAB & view:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    cap.release()


# In[ ]:


image_name="./test-data/IMG_5710_S.JPG"
#image_name="./test-data/IMG_6109.JPG"
#image_name=os.path.join(DRIVE_ROOT,"img_interviewMovie.jpg")
#image_name=os.path.join(DRIVE_ROOT,"test-data/12.jpg")

#single_image(image_name,detect_only=False,debug=True)
image=cv2.imread(image_name)
detect_face_save(image)


# In[10]:


import time 
# name='gassou_15sec_small'
# name='gassou_full_small'
name='gassou_full_large'
video_name=os.path.join(DRIVE_ROOT,name+'.mp4')
out_name = name+'.avi'
start = time.time()
video_proc(video_name, output=out_name, capture=True, detect_only=False, view=True)
#video_proc(0, detect_only=False, view=True)
elapsed_time=time.time() - start
print('elapsed time: ',format_time(elapsed_time))


# In[ ]:


if IN_COLAB:
  file_name = os.path.join(OUT_DIR,out_name)
  #file_name ="/content/gdrive/My Drive/opencv/output/gassou_full_large/gassou_full_large_04m*.png"
  from google.colab import files
  files.download(file_name)


# In[ ]:


import glob
#image_names=glob.glob(os.path.join(OUT_DIR,'gassou_full_large/*04m27s*.*'))
#image_names=glob.glob(os.path.join(OUT_DIR,'gassou_full_large/*.*'))
image_names=[]
image_names.append(os.path.join(OUT_DIR,'gassou_full_large/gassou_full_large_03m14s068ms.png'))
image_names.append(os.path.join(OUT_DIR,'gassou_full_large/gassou_full_large_04m46s828ms.png'))
image_names.append(os.path.join(OUT_DIR,'gassou_full_large/gassou_full_large_06m23s966ms.png'))
print('count:',len(image_names))
#image_names_sub=image_names[51:60]
for image_name in image_names:
  print(os.path.basename(image_name))
  image=cv2.imread(image_name)
  cv2_imshow(image)


# In[ ]:





# In[ ]:





# In[ ]:




