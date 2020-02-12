import cv2
import os
import argparse
#os.listdir(directory)
# Opens the Video file
root = os.getcwd()
img_folder = root + r'/images'
vid_folder = root + r'/videos/'
input_folder = root + r'/input/'

def vid2frames():
    video = os.listdir(input_folder)[0]
    if video not in os.listdir(img_folder):
        video_path = input_folder + video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 100:
            print("The video is too short!")
            os.remove(video_path)
            return None
        #print(f'Processing video at: {video_path}')
        interval = frame_count // 100
        #print(frame_count)

    if not cap.isOpened():
        print('Unable to capture video!')
    else:
        class_dir = img_folder + '/' + video
        os.mkdir(class_dir)
        print(f'New class folder created: {class_dir}')
        os.chdir(class_dir)
        i = 1
        saves = 0
        while(cap.isOpened() and saves<100):
            ret, frame = cap.read()
            if ret == False:
              break
            if i % interval == 0:
              cv2.imwrite(str(video) + '_' + str(saves) + '.jpg', frame)
              saves += 1
            i+=1
        cap.release()
        cv2.destroyAllWindows()
        os.chdir(root)
        print('-Images succesfully saved!-')
    os.remove(video_path)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Turning video into frames')
    #parser.add_argument('class_name', help='Name of the class')
    #args = parser.parse_args()

    vid2frames()
