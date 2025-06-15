# Importing all necessary libraries 
import cv2 
import os 

# Read the video from specified path 
cam = cv2.VideoCapture("data/football_test\Match_1953_2_0_subclip\Match_1953_2_0_subclip.mp4") 
output_folder = r"frame"
try: 
	
	# creating a folder named data 
	os.makedirs(output_folder, exist_ok=True)

# if not created then raise error 
except OSError: 
	print ('Error: Creating directory of data') 

# frame 
currentframe = 300
max_frames = 350
while currentframe < max_frames:
    ret, frame = cam.read()
    
    if not ret:
        print("error")
        break 

    # save frame
    frame_path = os.path.join(output_folder, f'frame{currentframe}.jpg')
    print(f'creating... {frame_path}')
    cv2.imwrite(frame_path, frame)

    currentframe += 1

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 

