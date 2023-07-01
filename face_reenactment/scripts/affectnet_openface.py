import os, sys
import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    TRAIN_DF = "affectnet_manually/Manually_Annotated_file_lists/training.csv"
    VAL_DF = "affectnet_manually/Manually_Annotated_file_lists/validation.csv"
    DIR_SRC = "affectnet_manually/"
    DIR_OUT = "affectnet/train_openface/"
    
    command = "software/OpenFace/build/bin/FaceLandmarkImg -f {img_path} -out_dir {out_dir} -aus -simalign -au_static -nobadaligned -simsize 128 -format_aligned jpg -nomask"
    
    train_df = pd.read_csv(TRAIN_DF)
    train_df = train_df[train_df['expression'].isin([0,1,2,3,4,5,6,7])]
    train_df.index = range(len(train_df))
    
    for i in range(len(train_df)):
        current_path = os.path.join(DIR_SRC, train_df.iloc[i]['subDirectory_filePath'])
        print(f"Current row: {i} | Current path: {current_path}")
        os.system(command.format(img_path=current_path, out_dir=DIR_OUT))