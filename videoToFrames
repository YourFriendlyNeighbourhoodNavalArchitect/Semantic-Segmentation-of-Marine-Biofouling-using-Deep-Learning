import cv2
import os

def videoToFrames(pathIn, pathOut, frameRate):
    # Simple function that cuts given video into its frames.
    # Portion of the dataset is in the form of video footage.
    # A lower frameRate value corresponds to more snapshots.
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 * frameRate))
        success, image = vidcap.read()
        if success:
            print(f'Read a new frame at {count * frameRate} seconds.')
            cv2.imwrite(os.path.join(pathOut, f'frame{count}.jpg'), image)
            count += 1
        else:
            print('End of video or error encountered.')

pathIn = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\Dataset\Videos\Cage\Cage.MP4'
# New images are saved in the same path as the original video.
pathOut = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\Dataset\Videos\Cage'
videoToFrames(pathIn, pathOut, 1)
