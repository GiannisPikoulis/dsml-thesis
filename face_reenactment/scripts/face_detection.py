from mtcnn import MTCNN
import cv2
import pandas as pd
import os

PATH = 'affectnet_manually/'

df = pd.read_csv('affectnet_manually/Manually_Annotated_file_lists/training.csv')
df = df[df['expression'].isin([0,1,2,3,4,5,6,7])]

detector = MTCNN()

OUTPATH = 'affectnet/train_cropped'

for i in range(114000, len(df)):
    row = df.iloc[i]
    img = cv2.cvtColor(cv2.imread(os.path.join(PATH, row['subDirectory_filePath'])), cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img)
    if len(detections) > 0:
        box = detections[0]['box']
        cropped_img = img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
        filename = os.path.join(OUTPATH, str(row['expression']) + '_' + row['subDirectory_filePath'].split('/')[1])
        print(f'Row: {i}, Filename: {filename}')
        cv2.imwrite(filename, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
    else:
        continue
        