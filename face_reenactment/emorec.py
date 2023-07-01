import os, sys
import numpy as np
import cv2
import torch
import pandas as pd
from hsemotion.facial_emotions import HSEmotionRecognizer


def run_emorec(dir0):
    
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    model_name = 'enet_b2_7'

    fer = HSEmotionRecognizer(model_name=model_name, device=device)
    
    # directory
    samples_dir = str(dir0)
    y_trg = int(samples_dir.split('/')[5])  
    sample_paths = sorted([os.path.join(samples_dir, f) for f in os.listdir(samples_dir)])

    emotion2idx_dict = {"Neutral": 0, "Happiness": 1, "Sadness": 2, "Surprise": 3, "Fear": 4, "Disgust": 5, "Anger": 6}
    idx2emotion_dict = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger"}

    counts = dict.fromkeys(emotion2idx_dict.keys(), 0)
    for i, path in enumerate(sample_paths):
        # print(f'Index: {i}, Path: {path}')
        frame_bgr = cv2.imread(path)
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(frame, logits=True)
        counts[emotion] += 1

    # print(f'Target emotion: {y_trg}')
    # print(f'Predictions: {counts}')
    
    for k in counts.keys():
        counts[k] = np.round(counts[k] / len(sample_paths), 3)
        
    return counts[idx2emotion_dict[y_trg]], counts, y_trg


def run_emorec_default(dir0):
    
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    model_name = 'enet_b2_8'

    fer = HSEmotionRecognizer(model_name=model_name, device=device)
    
    # directory
    samples_dir = str(dir0)
    y_trg = int(samples_dir.split('/')[5])  
    sample_paths = sorted([os.path.join(samples_dir, f) for f in os.listdir(samples_dir)])

    emotion2idx_dict = {"Neutral": 0, "Happiness": 1, "Sadness": 2, "Surprise": 3, "Fear": 4, "Disgust": 5, "Anger": 6, "Contempt": 7}
    idx2emotion_dict = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}

    counts = dict.fromkeys(emotion2idx_dict.keys(), 0)
    for i, path in enumerate(sample_paths):
        # print(f'Index: {i}, Path: {path}')
        frame_bgr = cv2.imread(path)
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(frame, logits=True)
        counts[emotion] += 1

    # print(f'Target emotion: {y_trg}')
    # print(f'Predictions: {counts}')
    
    for k in counts.keys():
        counts[k] = np.round(counts[k] / len(sample_paths), 3)
        
    return counts[idx2emotion_dict[y_trg]], counts, y_trg


def run_top2_emorec(dir0):
    
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    model_name = 'enet_b2_7'

    fer = HSEmotionRecognizer(model_name=model_name, device=device)
    
    # directory
    samples_dir = str(dir0)
    y_trg = int(samples_dir.split('emotion=')[1].split('_')[0]) 
    sample_paths = sorted([os.path.join(samples_dir, f) for f in os.listdir(samples_dir)])

    idx2emotion_dict = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger"}
    idx_to_class = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
    y_label = idx2emotion_dict[y_trg]
    
    count = 0
    for i, path in enumerate(sample_paths):
        # print(f'Index: {i}, Path: {path}')
        frame_bgr = cv2.imread(path)
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(frame)
        top2_indices = np.argsort(-scores)[:2]
        top2_emotions = [idx_to_class[ind] for ind in top2_indices]
        if y_label in top2_emotions:
            count += 1

#     print(f'Target emotion: {y_trg}')
#     print(f'Predictions: {counts}')
    
    count = np.round(count / len(sample_paths), 3)
        
    return count, y_trg


if __name__ == "__main__":
    
#     manipulated_emotion_dir_paths = sorted([os.path.join('samples/manipulations', DIR) for DIR in os.listdir('samples/manipulations') if DIR.isdigit()])
#     manipulated_emotion_dir_paths = sorted([os.path.join('samples/tuned_manipulations', DIR) for DIR in os.listdir('samples/tuned_manipulations') if DIR.isdigit()])

#     steps_list = []
#     scale_list = []
#     strength_list = []
#     emotion_list = []
#     predictions_list = []
#     target_percentage_list = []
#     lambda_dir_list = []

#     for emodir in manipulated_emotion_dir_paths:

#         dir_xrecs = sorted([os.path.join(emodir, f) for f in os.listdir(emodir) if os.path.isdir(os.path.join(emodir, f)) and 'xrec' in f])
#         dir_x0 = sorted([os.path.join(emodir, f) for f in os.listdir(emodir) if os.path.isdir(os.path.join(emodir, f)) and 'x0' in f])[0]

#         for dir_xrec in dir_xrecs:
#             target_percentage, y_trg = run_top2_emorec(dir_xrec)

#             steps = dir_xrec.split('_')[3]
#             steps = int(steps.split('ddim')[1])
#             scale, strength, emotion = dir_xrec.split('_')[5:8]
#             scale = float(scale.split('scale')[1])
#             strength = float(strength.split('strength')[1])
#             emotion = int(emotion.split('emotion=')[1])
            
#             if len(dir_xrec.split('_')) == 16 or len(dir_xrec.split('_')) == 13:
#                 steps = dir_xrec.split('_')[4]
#                 steps = int(steps.split('ddim')[1])
#                 scale, strength, emotion = dir_xrec.split('_')[6:9]
#                 scale = float(scale.split('scale')[1])
#                 strength = float(strength.split('strength')[1])
#                 emotion = int(emotion.split('emotion=')[1])
#                 if scale == 1:
#                     lambda_dir = 3
#                 else:
#                     lambda_dir = 2

#             elif len(dir_xrec.split('_')) == 19:
#                 steps = dir_xrec.split('_')[4]
#                 steps = int(steps.split('ddim')[1])
#                 scale, strength, emotion = dir_xrec.split('_')[6:9]
#                 scale = float(scale.split('scale')[1])
#                 strength = float(strength.split('strength')[1])
#                 emotion = int(emotion.split('emotion=')[1])
#                 lambda_dir = int(dir_xrec.split('_')[11])

#             else:
#                 raise NotImplementedError
            
#             assert y_trg == emotion

#             steps_list.append(steps)
#             scale_list.append(scale)
#             strength_list.append(strength)
#             emotion_list.append(emotion)
#             predictions_list.append(predicted)
#             target_percentage_list.append(target_percentage)
#             lambda_dir_list.append(lambda_dir)
            
#             print('lambda_dir: {}, DDIM Steps: {}, Scale: {}, Strength: {}, Label: {}, Target: {}'.format(lambda_dir, steps, scale, strength, emotion, target_percentage))

#     df = pd.DataFrame({'lambda_dir': pd.Series(lambda_dir_list), 'steps': pd.Series(steps_list), 'scale': pd.Series(scale_list), 'strength': pd.Series(strength_list), 'emotion': pd.Series(emotion_list), 'emorec_percentage': pd.Series(target_percentage_list)})

#     df = pd.concat([df, pd.DataFrame(predictions_list)], axis=1)
#     df.to_csv('emorec_results/top2_untuned.csv')
#     df.to_csv('emorec_results/top2_tuned.csv')
        
    emodir =  'affectnet/val_aligned_v3_split/'
    dir0s = sorted([os.path.join(emodir, f) for f in os.listdir(emodir) if os.path.isdir(os.path.join(emodir, f)) and f.isdigit()])    
    predictions_list = []
    target_percentage_list = []
    emotion_list = []
    
    for dir0 in dir0s:
        print(dir0)
        target_percentage, predicted, y_trg = run_emorec_default(dir0)
        predictions_list.append(predicted)
        target_percentage_list.append(target_percentage) 
        emotion_list.append(y_trg)
        print('Label: {}, Target: {}, All: {}'.format(y_trg, target_percentage, predicted))
        
df = pd.DataFrame({'emotion': pd.Series(emotion_list), 'emorec_percentage': pd.Series(target_percentage_list)})
df = pd.concat([df, pd.DataFrame(predictions_list)], axis=1)
df.to_csv('emorec_results/emorec_8_gt.csv')
    
#     dir0 =  'affectnet/val_aligned_v3/'
#     counts = run_emorec_default(dir0)
#     print(counts)

# use_cuda = torch.cuda.is_available()
# device = 'cuda' if use_cuda else 'cpu'
# model_name = 'enet_b2_7'

# fer = HSEmotionRecognizer(model_name=model_name, device=device)

# if os.path.exists('./emorec_results') and os.path.isdir('./emorec_results'):
#     print("Directory ./emorec_results already exists")
# else:
#     os.makedirs('./emorec_results', exist_ok=True)  
    
# # directory
# samples_dir = str(sys.argv[1])
# OUT_PATH = './emorec_results/' + samples_dir.split('/')[-1] + '.txt'
# f = open(OUT_PATH, 'w')  
# y_trg = int(samples_dir.split('emotion=')[1].split('_')[0])  
# sample_paths = sorted([os.path.join(samples_dir, f) for f in os.listdir(samples_dir)])

# emotion2idx_dict = {"Neutral": 0, "Happiness": 1, "Sadness": 2, "Surprise": 3, "Fear": 4, "Disgust": 5, "Anger": 6}
# idx2emotion_dict = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger"}

# counts = dict.fromkeys(emotion2idx_dict.keys(), 0)
# for i, path in enumerate(sample_paths):
#     print(f'Index: {i}, Path: {path}')
#     frame_bgr = cv2.imread(path)
#     frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#     emotion, scores = fer.predict_emotions(frame, logits=True)
#     counts[emotion] += 1

# print(f'Target emotion: {y_trg}')
# print(f'Predictions: {counts}')
# for k in counts.keys():
#     counts[k] = np.round(counts[k] / len(sample_paths), 3)
    
# f.writelines(f"Sample directory: {samples_dir}\n")
# f.writelines(f"Total images processed: {len(sample_paths)}\n")
# f.writelines(f"Predictions: {counts}\n")
# f.writelines(f"Target manipulated percentage: {counts[idx2emotion_dict[y_trg]]}")
# f.close()