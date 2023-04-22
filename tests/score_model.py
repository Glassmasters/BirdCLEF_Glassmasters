import pandas as pd
import sklearn.metrics


import gc

import pandas as pd
import pandas.api.types
import sklearn.metrics

from typing import Sequence, Union


class ParticipantVisibleError(Exception):
    pass


def padded_cmap(solution, submission, average, pos_label, sample_weight, padding_factor=5):
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average=average,
        pos_label=pos_label,
        sample_weight=sample_weight
    )
    return score


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, average: str='macro', pos_label: Union[int, str]=1, weights_column_name: str='weights', padding_factor: int=5) -> float:
    '''
    A variant of scikit-learn's cmAP.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score

    Adds N = padding_factor correct rows to both the submission and solution.
    This allows cmAP to support labels with zero true positives and reduces the effective impact of classes with a low count of positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    gc.collect()

    sample_weight = None
    if weights_column_name in solution.columns:
        sample_weight = solution.pop(weights_column_name).values
        if not pandas.api.types.is_numeric_dtype(sample_weight):
            raise ParticipantVisibleError('The solution weights are not numeric')
    if len(solution.columns) != len(submission.columns):
        raise ParticipantVisibleError('Invalid submission columns found')
    return padded_cmap(solution, submission, average=average,pos_label=pos_label,sample_weight=sample_weight, padding_factor=padding_factor)


if __name__ == '__main__':
    submission = pd.read_csv('../src/audio_to_image_model/submission.csv')

    solution = pd.read_csv('../tests/test_audio_metadata/all_mixed_audio_classes.csv')
    print(score(solution, submission, "row_id"))
