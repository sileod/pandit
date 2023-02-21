
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


def read_tsv(**kwargs):
    return pd.read_csv(**kwargs,sep='\t')

def read_jsonl(**kwargs):
    return pd.read_json(**kwargs,lines=True)

def read_auto(path, **kwargs):
    if path.endswith('.csv'):
        return pd.read_csv(path, **kwargs)
    if path.endswith('.tsv'):
        return read_tsv(path, **kwargs)
    if path.endswith('.json'):
        return pd.read_json(path, **kwargs)
    if path.endswith('.jsonl'):
        return read_jsonl(path, **kwargs)

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
    return pd.concat([name_df, config_df,summary_df], axis=1)


def sieve(df,d=dict(), **kwargs):
    df=df.copy()
    for k,v in {**d, **kwargs}.items():
        if type(v)!=list:
            v=[v]
        df=df[df[k].map(lambda x:x in v)]
    return df

def drop_constant(df):
    df=df.copy()
    return df.loc[:,df.astype(str).nunique()!=1]
    

def show(df,n=20,random=False,escape=True,sep_width=120):
    '''Aesthethic visualization of data with multiple (possibly long) text fields)'''
    df=df.copy()
    length=len(df)
    if random: 
        df=df.sample(frac=1.0)
    df=df.head(n)
    
    if hasattr(df,'columns'):
        for c in df.columns:
            df[c]='•'+df[c].map(str).map(str.strip)
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
pd.read_wandb = read_wandb
pd.read_jsonl = read_jsonl
pd.read_tsv = read_tsv
pd.read_auto = read_auto
PandasObject.drop_constant=drop_constant
PandasObject.sieve=sieve
