import cv2
import os

def videoToFrames(pathIn, pathOut, frameRate):
    # Simple function that cuts given video into its frames.
    # Portion of the dataset is in the form of video footage.
    # A lower frameRate value corresponds to more snapshots.
    count = 0
    video = cv2.VideoCapture(pathIn)
    success, image = video.read()
    success = True
    while success:
        video.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 * frameRate))
        success, image = video.read()
        if success:
            print(f'Read a new frame at {count * frameRate} seconds.')
            cv2.imwrite(os.path.join(pathOut, f'frame{count}.jpg'), image)
            count += 1
        else:
            print('End of video or error encountered.')

pathIn = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\Videos\Cage\Cage.MP4'
# New images are saved in the same path as the original video.
pathOut = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\Videos\Cage'
frameRate = 1
videoToFrames(pathIn, pathOut, frameRate)
