from PIL import Image
import numpy as np
from utils import RipCurrentDataConf


def load_rip_annotation(conf: RipCurrentDataConf, video_name: str) -> np.ndarray:

    file = conf.annotation_path / conf.annotation_filename.format(video_name=video_name)
    img = Image.open(file)
    return (np.asarray(img) / 255.).astype(int)


def load_estimated_rip(conf: RipCurrentDataConf, video_name: str, algorithm_name: str) -> np.ndarray:
    file = conf.result_path / conf.heatmap_filename.format(video_name=video_name,
                                                           algorithm_name=algorithm_name)

    data = np.load(str(file))['arr_0'].astype(int)
    return data.transpose()
