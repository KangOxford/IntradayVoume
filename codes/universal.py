import pandas as pd
def get_universal_df():
    df_lst = []
    from tqdm import tqdm
    for i in tqdm(range(start_index, start_index + num)):  # on mac4
        df = pd.read_pickle(path0600_1 + path0600_1Files[i])
        df_lst.append(df)

    new_dflst_lst = []
    for index, dflst in enumerate(df_lst):
        # assert dflst.shape[0] == 3172, f"index, {index}"
        if dflst.shape[0] == 3146:
            new_dflst_lst.append(dflst)

    gs = [dflst.iterrows() for dflst in new_dflst_lst]
    dff = []
    for i in tqdm(range(dflst.shape[0])):
        for g in gs:
            elem = next(g)[1].T
            dff.append(elem)
    df = pd.concat(dff, axis=1).T
    df.reset_index(inplace=True, drop=True)
    return df
