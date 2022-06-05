## Strategy
We will experiment with four models as follows
1. Based on manually extracted features using librosa - will be done by <b>Sanjeev</b>
2. Based on feature extraction from wave2vec - will be done by <b>Sanjeev</b>
3. Based on feature extraction from HuBert - will be done by <b>Pragnya</b>
4. Based on feature extraction from EmBert - will be done by <b>Reece</b>

## Approach

Update as of June 4th, 2022
#### Step 1 : Data prepration
code: data_prep.py \
We are doing the following: \
    1. Converting all m4a files to wav \
    2. replacing m4a to wav in the labeled json data \
    3. Capturing duration of each sound file \
    4. We made all the audio files of equal length by adding segments in the front and back of each soundfile.
    We did this intially by adding Whitnoise but later changed to silence. Final length of each file is 106 sec. \


#### Step 2a : Feature Extraction using Librosa
code: spectfiles.py \
This program does the following:
    1. Extracts Mels from each audio files
    2. Run parallel jobs to save Mels for each audio file in it own pickle file
    
#### Step 2b : Feature Extraction using wav2vec2
code: wave3vecfeatures.py \
This program does the following:\
    1. Extract features from each audio files for each transformer layer in wave2 vec2. There are 12 transformer layers and we have extracted features for all for each file in pickle format. Each file is 500 MB.\
    2. Run parallel jobs to save feature for each audio file in it own pickle file

#### Step 3 : Feature consolidation
code: features.py:
This file does the following
1. Consolidate the multiple feature pickle files (We did it to process in parallel) into one feature file one for librosa and one for wave2vec \
2. Create labeled dataset by each therapist and emotion. For example: Yared Alemu_fear.csv. In other words we made this binary classification problem \
3. We also had to balance the data in the sense that for each classification we have have equal number of data points \

#### Step 4 : Model building using Librosa
code: librosaModel.ipynb \
We fed in librosa features and some basic three layer conv2d to check if results are marginally better than mere chance. We got 54% accuracy. We were hoping for around 60%.
