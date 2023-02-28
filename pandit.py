from pandas import *
import pandas as pd
import warnings
import copy
import numpy as np
from collections import Iterable
from pandas.core.base import PandasObject
from IPython.display import display, HTML
import html
from dataclasses import dataclass
from string import Template

def read_tsv(*args,**kwargs):
    return pd.read_csv(*args,**kwargs,sep='\t')

def to_tsv(df, *args,**kwargs):
    return df.to_csv(*args,**kwargs,sep='\t')

def read_jsonl(*args,**kwargs):
    return pd.read_json(*args,**kwargs,lines=True)

def to_jsonl(df, *args,**kwargs):
    return df.to_json(*args,**kwargs,lines=True, orient='records')

def read(path, *args, **kwargs):
    end=None
    try:
        end = path.split('.')[-1]
        getattr(pd,f'read_{end}')(path,*args,**kwargs)
    except:
        print(f'Format could not be guessed based on the extension. ({end})')
        return None

def read_wandb(project_name, exclude_gradients=True):
    """source: https://docs.wandb.ai/guides/track/public-api-guide"""
    import wandb
    api = wandb.Api(timeout=49)
    runs = api.runs(project_name)
    summary_list = [] 
    config_list = [] 
    name_list = []
    condition = lambda x: ('gradients/' not in x) if exclude_gradients else True
    for run in runs: 
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
        summary_list.append({k:v for k,v in run.summary._json_dict.items() if condition(k)}) 
        # run.config is the input metrics.  We remove special values that start with _
        config_list.append({k:v for k,v in run.config.items() if not ('hash' not in k and k.startswith('_'))}) 
        # run.name is the name of the run.
        name_list.append(run.name)       
    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list}) 
    df = pd.concat([name_df, config_df,summary_df], axis=1)
    df=df.loc[:,~df.columns.duplicated()]   
    return df

def sieve(df,d=dict(), **kwargs):
    df=df.copy()
    #df=df.loc[:,~df.columns.duplicated()]
    for k,v in {**d, **kwargs}.items():
        if type(v)!=list:
            v=[v]
        df=df[df[k].map(lambda x:x in v)]
    return df

def drop_constant_columns(df):
    df=df.copy()
    return df.loc[:,df.astype(str).nunique()!=1]


def _bold_row(x):
    if x.dtype==np.float64: 
        x=x.map(lambda c: f"$\textbf{{{round(c,1)}}}$" if c==x.max() else "{:.1f}".format(c))
    return x

def bold_max(df):
    return df.apply(_bold_row)

def to_dropbox(df, path, format=None, token=None,**kwargs):
    import dropbox
    if not format:
        format=path.split('.')[-1]
    dbx = dropbox.Dropbox(token)
    df_string = getattr(df,f'to_{format}')(**kwargs)
    db_bytes = bytes(df_string, 'utf8')
    dbx.files_upload(
        f=db_bytes,
        path=path,
        mode=dropbox.files.WriteMode.overwrite
    )

def show(df,n=20,random=False,escape=True,sep_width=120):
    '''Aesthethic visualization of data with multiple (possibly long) text fields)'''
    df=df.copy()
    length=len(df)
    if random: 
        df=df.sample(frac=1.0)
    df=df.head(n)
    
    if hasattr(df,'columns'):
        for c in df.columns:
            df[c]='─ '+df[c].map(str).map(str.strip)
    df.index=['─'*sep_width]*len(df)

    s=df.to_csv(None,sep="\n")
    if escape:
        s=html.escape(s)
    s=f'length:{length}\n{s}'.replace('\n','<br>')
    return HTML(f'<font face="Arial" size="2px">{s}</font>')

def rshow(df,n=20):
    '''Aesthethic visualization of shuffled data with multiple (possibly long) text fields)'''
    return show(df,n,random=True)


PandasObject.show = show
PandasObject.rshow = rshow
PandasObject.bold_max = bold_max

pd.read_wandb = read_wandb
pd.read_jsonl = read_jsonl
pd.read_tsv = read_tsv
pd.read = read
PandasObject.drop_constant=drop_constant_columns
PandasObject.sieve=sieve

PandasObject.to_dropbox=to_dropbox
PandasObject.to_jsonl=to_jsonl
PandasObject.to_tsv=to_tsv
