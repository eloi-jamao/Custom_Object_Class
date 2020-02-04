import cv2
import os
import argparse
#os.listdir(directory)
# Opens the Video file
def vid2frames():
  #for video in vid_folder:
  root = os.getcwd()
  img_folder = root + '/images'
  vid_folder = root + '/videos/'
  for video in os.listdir(vid_folder):
    if video not in os.listdir(img_folder):
      video_path = root + '/videos/' + video # + '.mp4'
      print(f'Processing video at: {video_path}')
      cap = cv2.VideoCapture(video_path)

      frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      if frame_count < 100:
        print("The video is too short!")

      interval = frame_count // 100
      #print(frame_count)

      if not cap.isOpened():
        print('Unable to capture video!')
      else:
          class_dir = img_folder + '/' + video
          os.mkdir(class_dir)
          print(f'New class folder created: {class_dir}')
          os.chdir(class_dir)
          i = 0
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
          print('-----------Images succesfully saved!-----------')



if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Turning video into frames')
    #parser.add_argument('class_name', help='Name of the class')
    #args = parser.parse_args()
    vid2frames()
