
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, cohen_kappa_score, accuracy_score, precision_recall_curve, auc, roc_auc_score, average_precision_score
from sklearn.neighbors import KernelDensity

from brainmaze_eeg.scikit_modules import ZScoreModule


def compare_datasets(dataset, states=['AWAKE', 'N2', 'N3', 'REM']):
    """
    Compares dataset consistency using KDE and prec-recall curves

    Parameters
    ----------
    dataset: dict
        dict where the key is name of dataset for comparison; each dataset is represented by dict with 2 variables X, Y

    Returns
    -------
    dataset: dict
    """
    # AUPRC, AUROC, likelihood
    # sturcture
    # dataset_name - measure_name: AUPRC, AUROC, av likelihood_per_class, av_likelihood, class_frequency - class

    scores = {}
    for idx1, k1 in enumerate(dataset.keys()):
        for idx2, k2 in enumerate(dataset.keys()):
            x1 = dataset[k1]['X']
            y1 = dataset[k1]['Y']
            x2 = dataset[k2]['X']
            y2 = dataset[k2]['Y']

            #nm = k1 + ' - ' + k2

            ZS = ZScoreModule(trainable=True)
            x1 = ZS.fit_transform(x1)
            x2 = ZS.transform(x2)

            kde = KernelDensity().fit(x1)
            #y1_ = kde.score_samples(x1)
            y2_ = kde.score_samples(x2)

            if not k1 in scores.keys(): scores[k1] = {}

            if not k2 in scores[k1].keys():
                scores[k1][k2] = {}
                scores[k1][k2]['loglikelihood'] = {}
                scores[k1][k2]['auroc'] = {}
                #scores[k1][k2]['auprc'] = {}
                scores[k1][k2]['ap'] = {}

            scores[k1][k2]['loglikelihood']['all'] = y2_.mean()

            for cl in states:
                kde = KernelDensity().fit(x1[y1 == cl])
                sc = kde.score_samples(x2)
                #pr, rc, thresholds = precision_recall_curve(y2 == cl, sc)
                #pr = pr[1:]
                #rc = rc[1:]
                #auprc = auc(rc, pr)
                auroc = roc_auc_score(y2 == cl, sc)
                ap = average_precision_score(y2 == cl, sc)
                #scores[k1][k2]['auprc'][cl] = auprc
                scores[k1][k2]['auroc'][cl] = auroc
                scores[k1][k2]['ap'][cl] = ap
                scores[k1][k2]['loglikelihood'][cl] = sc.mean()
    return scores