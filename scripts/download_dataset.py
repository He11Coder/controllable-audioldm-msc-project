"""
This script downloads AudioSet dataset to your local machine. See https://pypi.org/project/audioset-download/ for details.
"""

import os
from audioset_download import Downloader
from pathlib import Path

# Number of CPUs available
CPU_COUNT = os.cpu_count()

# Target directory
DATASET_FOLDER = Path("/scratch/vb00479/audioset_balanced_22k")
TRAIN_FOLDER = DATASET_FOLDER.joinpath("train")
TRAIN_FOLDER.mkdir(parents=True, exist_ok=True)

# Select dataset type by changing `download_type`. See the docs for more info on the parameters
d = Downloader(root_path=TRAIN_FOLDER, n_jobs=CPU_COUNT*8, download_type='unbalanced_train', copy_and_replicate=False)

print("Download started")
# This will download a balanced version of AudioSet (~22k data points/~30GB of disk space)
d.download(format='wav')
print("Download complete")