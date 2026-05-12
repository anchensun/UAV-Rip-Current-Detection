from typing import List, Dict, Tuple, Any
import numpy as np
from data import load_rip_annotation, load_estimated_rip
import matplotlib.pyplot as plt
from utils import RipCurrentDataConf
plt.rcParams.update({'font.size': 16})


def precision_recall_curve(ref: np.ndarray,
                           pred: np.ndarray,
                           eps: float = 1e-8) -> Tuple[List[float], List[float], List[float], float]:
    """ Compute the precision recall curve of the given rip current region segmentation result

    :param ref: reference result of rip current regions
    :param pred: estimated rip current regions
    :param eps: small values to facilitate precision recall computation by avoiding 0./0.
    :return: threshold, precision and recall values
    """
    if ref.shape != pred.shape:
        raise AttributeError('The reference and estimated results do not have the same shape')

    thresholds = list()
    precisions = list()
    recalls = list()
    auc = 0.
    for th in range(0, np.max(pred)):
        mask = pred >= th
        tp = (ref & mask).sum()
        fp = (~ref & mask).sum()
        fn = (ref & ~mask).sum()
        thresholds.append(th)
        precisions.append(tp / (tp + fp + eps))
        recalls.append(tp / (tp + fn + eps))
        if th != 0:
            assert recalls[-2] >= recalls[-1]
            auc += (precisions[-2] + 0.5 * abs(precisions[-2] - precisions[-1])) * (recalls[-2] - recalls[-1])

    return thresholds, precisions, recalls, auc


def evaluate(conf: RipCurrentDataConf,
             video_names: List[str] or str,
             algorithm_names: List[str] or str,
             metrics: List[str] or str,
             merged: bool = False) -> Dict[str, Any]:

    if isinstance(video_names, str):
        video_names = [video_names]
    if isinstance(algorithm_names, str):
        algorithm_names = [algorithm_names]
    if isinstance(metrics, str):
        metrics = [metrics]

    results = dict()
    for video_name in video_names:
        annotation = load_rip_annotation(conf, video_name)
        if merged:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_ylabel('Precision')
            ax.set_xlabel('Recall')
            # ax.set_title('{}_Precision_Recall_Curve'.format(video_name))

        markers = ['o-', '^-', 's-', '*-', 'd-']
        for alg_idx, alg_name in enumerate(algorithm_names):
            # print(video_name, alg_name)
            estimation = load_estimated_rip(conf, video_name, alg_name)
            for metric in metrics:
                metric_func = globals()[metric]
                results[metric] = metric_func(annotation, estimation)
                # print(results[metric])

            if 'precision_recall_curve' in metrics:
                rst = results['precision_recall_curve']
                if not merged:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.set_ylabel('Precision')
                    ax.set_xlabel('Recall')
                    ax.set_title('{}_{}_PRC (AUC = {:.3})'.format(video_name, alg_name, rst[3]))

                ax.plot(rst[2], rst[1], markers[alg_idx], markevery=len(rst[1])//10)
                ax.set_xlim([-0.05, 1.05])
                ax.set_ylim([-0.05, 1.05])
                print('{}_{}_PRC (AUC = {:.3})'.format(video_name, alg_name, rst[3]))
                if not merged:
                    plt.savefig(conf.result_path / conf.prc_filename.format(video_name=video_name,
                                                                            algorithm_name=alg_name))

        if merged:
            algorithm_names_tmp = algorithm_names.copy()
            for idx, alg_name in enumerate(algorithm_names_tmp):
                algorithm_names_tmp[idx] = alg_name.replace('Schunk', 'Schunck').replace('FirstOrder-', '').replace('SecondOrder', 'HOR')
            plt.legend(algorithm_names_tmp)
            plt.savefig(conf.result_path / conf.prc_filename.format(video_name=video_name,
                                                                    algorithm_name='merged'))

    return results


if __name__ == '__main__':
    conf = RipCurrentDataConf('/home/bear-c/users/asun/RipCurrent/Near_Shore_Wave_Speed_Estimation/rip_curent_evaluation/workspace/conf.yaml')
    evaluate(conf, conf.video_list, conf.algorithm_list, 'precision_recall_curve', merged=True)
