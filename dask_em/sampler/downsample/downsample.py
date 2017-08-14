import string

import pandas as pd
from dask import threaded, delayed
from dask_em.sampler.downsample.dsprober import DownSampleProber
from dask import dataframe as dd

from dask_em.utils.cy_utils.stringcontainer import StringContainer
from dask_em.utils.py_utils.utils import get_str_cols, str2bytes, sample, split_df, \
    tokenize_strings_wsp, build_inv_index

# helper functions
def preprocess_table(dataframe, idcol):
    strcols = list(get_str_cols(dataframe))
    strcols.append(idcol)
    projdf = dataframe[strcols]
    objsc = StringContainer()
    for row in projdf.itertuples():
        colvalues = row[1:-1]
        uid = row[-1]
        strings = [colvalue.strip() for colvalue in colvalues if not pd.isnull(colvalue)]
        concat_row = str2bytes(' '.join(strings).lower())
        concat_row = concat_row.translate(None, string.punctuation)
        objsc.push_back(uid, concat_row)
    return objsc

def probe(objtc, objinvindex, yparam):
    objprobe = DownSampleProber()
    objprobe.probe(objtc, objinvindex, yparam)
    return objprobe

def lpostprocess(result_list):
    lids = set()
    for i in range(len(result_list)):
        result = result_list[i]
        lids.update(result.get_lids())
    lids = sorted(lids)
    return lids


def rpostprocess(result_list):
    rids = set()
    for i in range(len(result_list)):
        result = result_list[i]
        rids.update(result.get_rids())
    rids = sorted(rids)
    return rids


def select(table, ids):
    return table.loc[ids]

def get_meta(table):
    vals = []
    for col in table.columns:
        if table[col].dtype == 'int64':
            vals.append(0)
        elif table[col].dtype == 'float64':
            vals.append(0.0)
        elif table[col].dtype == 'bool':
            vals.append(False)
        elif table[col].dtype == 'object':
            vals.append('')
        else:
            raise ValueError(col)
    result = [tuple(vals)]
    meta = pd.DataFrame(result, columns=list(table.columns))
    return meta

def downsample_dd(ltable, rtable, lid, rid, fraction, y, lstopwords=[],
                  rstopwords=[]):
    ltokens = []
    for i in range(ltable.npartitions):
        lcat_strings = (delayed)(preprocess_table)(ltable.get_partition(i), lid)
        tokens = (delayed)(tokenize_strings_wsp)(lcat_strings, lstopwords)
        ltokens.append(tokens)

    invindex = (delayed)(build_inv_index)(ltokens)
    rsample = rtable.sample(fraction, random_state=0)
    rsample.set_index(rid)

    probe_rslts = []
    for i in range(rsample.npartitions):
        rcat_strings = (delayed)(preprocess_table)(rsample.get_partition(i), rid)
        rtokens = (delayed)(tokenize_strings_wsp)(rcat_strings, rstopwords)
        probe_rslt = (delayed)(probe)(rtokens, invindex, y)
        probe_rslts.append(probe_rslt)

    lresult = (delayed)(lpostprocess)(probe_rslts)
    rresult = (delayed)(rpostprocess)(probe_rslts)

    lmeta = get_meta(ltable)
    resA = []
    for i in range(ltable.npartitions):
        tmp = (delayed)(select)(ltable.get_partition(i), lresult)
        resA.append(tmp)
    A1 = dd.from_delayed(resA, meta=lmeta)

    resB = []
    rmeta = get_meta(rtable)
    for i in range(rtable.npartitions):
        tmp = (delayed)(select)(rtable.get_partition(i), rresult)
        resB.append(tmp)
    B1 = dd.from_delayed(resB, meta=rmeta)

    return A1, B1









