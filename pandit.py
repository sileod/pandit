from pandas import *
import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
from IPython.display import HTML
import html
import os
from importlib.util import spec_from_file_location, module_from_spec

credentials=None # 
"""
set with pd.credentials=credentials, where 
credentials.dropbox returns dropbox api key
gsheet is the credential dict in https://docs.gspread.org/en/latest/oauth2.html

"""
def get_credential(service, credential=None):
    if credential:
        return credential
    else:
        return getattr(credentials, service)

def read_tsv(*args,**kwargs):
    return pd.read_csv(*args,**kwargs,sep='\t')

def to_tsv(df, *args,**kwargs):
    return df.to_csv(*args,**kwargs,sep='\t')

def read_jsonl(*args,**kwargs):
    return pd.read_json(*args,**kwargs,lines=True)

def to_jsonl(df, *args,**kwargs):
    return df.to_json(*args,**kwargs,lines=True, orient='records')

def read_sheet(id):
    return pd.read_csv(f'https://docs.google.com/spreadsheets/d/{id}/export?format=csv')
    
def read(path, *args, **kwargs):
    end=None
    try:
        end = path.split('.')[-1]
        return getattr(pd,f'read_{end}')(path,*args,**kwargs)
    except:
        print(f'Format could not be guessed based on the extension. ({end})')
        return None

def to(df, path, *args, **kwargs):
    end=None
    try:
        end = path.split('.')[-1]
        return getattr(df,f'to_{end}')(path,*args,**kwargs)
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
        if callable(v):
            df=df[df[k].map(v)]
            continue
        if type(v)!=list:
            v=[v]
        df=df[df[k].map(lambda x:x in v)]
    return df

def drop_constant_columns(df):
    df=df.copy()
    return df.loc[:,df.astype(str).nunique()!=1]


def safe_sample(df,n=None,*args, **kwargs):
    if not n or len(df)<=n:
        return df
    else:
        return df.sample(n,*args,**kwargs)

def _bold_row(x):
    if x.dtype==np.float64: 
        x=x.map(lambda c: f"$\textbf{{{round(c,1)}}}$" if c==x.max() else "{:.1f}".format(c))
    return x

def bold_max(df):
    return df.apply(_bold_row)

def to_dropbox(df, path, format=None, token=None,**kwargs):
    credential=get_credential('dropbox',credential)
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

def to_sheets(df,id,sheet_name,credential=None, include_index=False):
    credential=get_credential('gsheet',credential)
    import gspread
    from gspread_dataframe import set_with_dataframe
    gc = gspread.service_account_from_dict(credential)
    sh = gc.open_by_key(id)
    set_with_dataframe(worksheet=sh.worksheet(sheet_name),
        dataframe=df,
        include_index=include_index)


def show(df,n=20,random=False,escape=True,sep_width=120,header=False,length=True):
    '''Aesthethic visualization of data with multiple (possibly long) text fields)'''
    df=df.copy()
    length=len(df)
    if random: 
        df=df.sample(frac=1.0)
    df=df.head(n)
    
    if hasattr(df,'columns'):
        for c in df.columns:
            df[c]='  '+df[c].map(str).map(str.strip)
    df.index=['─'*sep_width]*len(df)

    s=df.to_csv(None,sep="\n",header=header)
    if escape:
        s=html.escape(s)
    if length:
        s=f'length:{length}\n{s}'.replace('\n','<br>')
    return HTML(f'<font face="Arial" size="2px">{s}</font>')

def rshow(df,n=20):
    '''Aesthethic visualization of shuffled data with multiple (possibly long) text fields)'''
    return show(df,n,random=True)


def undersample(df, column='label',sampling_strategy='auto',random_state=None,replacement=False):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy,random_state=random_state,replacement=replacement)
    df, _=rus.fit_resample(df, list(df[column]))
    return df.sample(frac=1.0).reset_index(drop=True)


def explode_dict(df, column):
    df=df.reset_index(drop=True)
    dfc=pd.DataFrame(list(df[column]))
    df = pd.concat([df,dfc],axis=1)
    del df[column]
    return(df)

def train_validation_test_split(df, train_frac=0.8, val_test_frac=0.5):
    df['_random'] = sorted(np.linspace(0,1, len(df)), key=lambda x:random.random()) 
    def get_split(x):
        if x<train_frac: return "train"
        if x<train_frac+(1-train_frac)*val_test_frac: return "validation"
        return "test"
    df['_split'] = df['_random'].map(get_split)    
    del df['_random']
    return df

pd.read_wandb = read_wandb
pd.read_jsonl = read_jsonl
pd.read_tsv = read_tsv
pd.read = read

PandasObject.show = show
PandasObject.rshow = rshow
PandasObject.bold_max = bold_max
PandasObject.drop_constant_columns=drop_constant_columns
PandasObject.explode_dict = explode_dict
PandasObject.sieve=sieve
PandasObject.train_validation_test_split=train_validation_test_split
PandasObject.safe_sample=safe_sample
PandasObject.undersample=undersample
PandasObject.to_dropbox=to_dropbox
PandasObject.to_sheets=to_sheets
PandasObject.to_jsonl=to_jsonl
PandasObject.to_tsv=to_tsv
PandasObject.to=to
