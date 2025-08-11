import os
from pathlib import Path


cur_dir = os.getcwd()
data_dir = Path(cur_dir, 'data')
features_dir = Path(cur_dir, 'features')


train = False
audio_reconstruction=True