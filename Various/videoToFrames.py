from cv2 import CAP_PROP_POS_MSEC, VideoCapture, imwrite
from os.path import join
from configurationFile import ALL_PATH

def videoToFrames(pathIn, pathOut, frameRate):
    # Simple function that cuts given video into its frames.
    # Portion of the dataset is in the form of video footage.
    # A lower frameRate value corresponds to more snapshots.
    count = 0
    savedFrames = 0
    video = VideoCapture(pathIn)
    if not video.isOpened():
        print(f"Unable to open video file at {pathIn}.")
        return

    while True:
        video.set(CAP_PROP_POS_MSEC, (count * 1000 * frameRate))
        success, image = video.read()
        if not success:
            break

        imwrite(join(pathOut, f'frame{count}.jpg'), image)
        print(f"Read a new frame at {count * frameRate} seconds.")
        savedFrames += 1
        count += 1
    
    video.release()
    print(f"Video processing complete. {savedFrames} frames saved to {pathOut}.")
      
pathIn = join(ALL_PATH, r'Videos\Cage.mp4')
pathOut = join(ALL_PATH, 'Videos')
frameRate = 0.5
videoToFrames(pathIn, pathOut, frameRate)