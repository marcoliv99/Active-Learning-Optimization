import pandas as pd
import os
import warnings
from typing import Literal


def load_dataset(df: pd.DataFrame | str | os.PathLike, 
                 path_labels: str | os.PathLike,
                 threshold_hit: float, 
                 on_missing: Literal['raise', 'drop', 'zero'] = 'raise') -> pd.DataFrame:
    """Merge the round predictions with the experimental labels.
    Args:
        df: round predictions, as a DataFrame or as the path to a csv file.
            Must contain the columns 'variant' and 'y_pred'.
        path_labels: path to the csv file with the experimental data.
            Must contain the columns 'variant' and 'activity'.
        threshold_hit: activity value from which a variant counts as a hit.      
        on_missing: what to do with variants of df that are not in the labels:
            'raise' (default) stops with an error; 
            'drop' discards them with a warning;
            'zero' keeps them as inactive (activity 0, last true_rank) with a warning.

    Returns:
        DataFrame sorted by y_pred (descending) with the columns of df +
        'activity', 'true_rank' (rank of the variant by true activity, 1 = best)
        and 'activity_binary' (1 if activity >= threshold_hit, else 0).
    """
    #Managing inputs
    if str(path_labels).endswith(('.xlsx', '.xls')):
        raise RuntimeError('path_labels is an excel file, please use a csv format')
    if isinstance(df, pd.DataFrame):
        df_round = df.copy()
    elif isinstance(df, (str, os.PathLike)):
        if str(df).endswith(('.xlsx', '.xls')):
            raise RuntimeError('df is an excel file, please use a csv format')
        df_round = pd.read_csv(df)
    else:
        raise TypeError(f'df must be a DataFrame or the path to a csv file, got {type(df).__name__}')
    if on_missing not in ('raise', 'drop', 'zero'):
        raise ValueError(f"on_missing must be 'raise', 'drop' or 'zero', got {on_missing!r}")

    df_labels = pd.read_csv(path_labels)
    df_labels_sorted = df_labels.sort_values(by='activity', ascending=False).reset_index(drop=True)
    df_labels_sorted["true_rank"] = df_labels_sorted.index+1
    labels_columns = df_labels_sorted[['variant', 'activity', 'true_rank']]
    df_merged = pd.merge(df_round, labels_columns, on='variant', how='left')

    #Managing NaN
    unmatched = df_merged['activity'].isna()
    if unmatched.any():
        missing = df_merged.loc[unmatched, 'variant'].tolist()
        msg = (f'{len(missing)}/{len(df_merged)} variants have no label, {missing[:10]}' 
               f'(match rate {1 - len(missing) / len(df_merged):.1%})')
        if on_missing == 'raise':
            raise ValueError(msg + ". Use on_missing = 'drop' or 'zero' to keep going")
        if on_missing == 'drop':
            warnings.warn(msg + ': dropped')
            df_merged = df_merged[~unmatched]
        else:
            warnings.warn(msg + ': treated as inactive')
            df_merged.loc[unmatched, 'activity'] = 0
            df_merged.loc[unmatched, 'true_rank'] = len(df_labels_sorted) + 1
    df_sorted = df_merged.sort_values(by='y_pred', ascending=False).reset_index(drop=True)
    df_sorted['activity_binary'] = (df_sorted['activity'] >= threshold_hit).astype(int)
    return df_sorted

#Create a dataframe in which store the results
def store_results(path: str,  
                  n_rounds: int):
    """This creates a nested dictionary with the replicates and the rounds for each replicate,
    storing the df_sorted_all.csv file within it. The dict is create from the directory hiercarchy 
    in which the results are stored. The input dict must be empty. 
    Please note that the n_rounds value must be written as n_rounds+1 otherwise
    you will lose the last round
    Args: path --> directory and subdirectories where results are saved
    protein_name 
    n_rounds --> Number of rounds done with Evolve
    Return: dictionary with the same structure of the dir and sub dir"""
    d = {}
    #iterate over dir in path given in input 
    for dir in os.listdir(path):
        #for iterating over the dir
        if dir.endswith('_rep'): 
            #empty dict for storing
            d[dir] = {} 
            for i in range(1, n_rounds): #loop for iterate over rounds
                if path.endswith(('.xlsx', '.xls')):
                    raise RuntimeError(f'The uploaded Dataframe is an excel file, please use a csv format')
                rep_path = os.path.join(path, dir, f'Round{i}')
                d[dir][f'Round{i}']= None
                round_path = os.listdir(rep_path)
                file_path = os.path.join(rep_path, 'df_sorted_all.csv')
                d[dir][f'Round{i}']= pd.read_csv(file_path)
        else:
            raise RuntimeError(f'No directory ends with _rep, please rename your directory')
    return d

def create_round_file(df_labels: pd.DataFrame,
                      voi: list) -> pd.DataFrame:
    """Read and process data from Labels df and voi(variant of interest) list
    This file is used as input in EvolvePro Rounds
    Args:
    df_labels--> Dataframe from the DMS experiments
    voi-->List containing the variant of interest that the round of Evolve predicted 
    Return: A dataframe with 2 columns: one for the variant and 
    one with the activity from labels file"""
    d = {}
    for v,a in zip(df_labels['Variant'], df_labels['activity']):
        if v in voi:
            v_short = v[1:]
            d[v_short] = a
    df = pd.DataFrame.from_dict(d, orient='index')
    df.index.name = 'Variant'
    df.columns = ['activity']
    df = df.reset_index()
    return df