import cv2
import os
import argparse
#os.listdir(directory)
# Opens the Video file
def vid2frames(class_name):
    #for video in vid_folder:
    root = os.getcwd()
    img_folder = root + '/images'
    video_path = root + '/videos/' + class_name + '.mp4'
    print(f'Processing video at: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Unable to capture video!')
    else:
        class_dir = img_folder + '/' + class_name
        os.mkdir(class_dir)
        print(f'New class folder created: {class_dir}')
        os.chdir(class_dir)
        i = 0
        saves = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if i % 5 == 0:
                cv2.imwrite(str(class_name) + '_' + str(saves) + '.jpg', frame)
                saves += 1
            i+=1

        cap.release()
        cv2.destroyAllWindows()
        os.chdir(root)
        print('-----------Images succesfully saved!-----------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turning video into frames')
    parser.add_argument('class_name', help='Name of the class')
    args = parser.parse_args()
    vid2frames(args.class_name)
