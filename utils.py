import pandas as pd
import os


def load_dataset(df:pd.DataFrame | dict, path_labels: str, 
                 threshold_hit: float)-> pd.DataFrame:
    """Loads two dataframes: one from the dms/experimental data and one from the round, 
    merging the column activity from the first one into the second and calculating
    a new column that contains the activity in binary form based on a threshold. 
    Please note that the path argument can be both a str or a directed pd.DataFrame """
    
    if path_labels.endswith(('.xlsx', '.xls')):
        raise RuntimeError(f'The uploaded Dataframe is an excel file, please use a csv format')
    if isinstance(df,str):
        df_round = pd.read_csv(df)
        if df.endswith(('.xlsx', '.xls')):
            raise RuntimeError(f'The uploaded Dataframe is an excel file, please use a csv format')
    elif isinstance(df, pd.DataFrame):
        df_round = df.copy()
    df_labels = pd.read_csv(path_labels)
    df_labels_sorted = df_labels.sort_values(by='activity', ascending=False).reset_index(drop=True)
    df_labels_sorted["true_rank"] = df_labels_sorted.index+1
    print(df_labels_sorted.head(10))
    labels_activity = df_labels_sorted[['variant', 'activity', 'true_rank']]
    df_merged = pd.merge(df_round, labels_activity, on='variant', how='left')
    df_clean = df_merged[df_merged['activity'].notna()]
    df_ordinato = df_clean.sort_values(by='y_pred', ascending=False).reset_index(drop=True)
    df_ordinato['activity_binary'] = (df_ordinato['activity'] >= threshold_hit).astype(int)
    print(df_ordinato.head(10))
    
    return df_ordinato

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