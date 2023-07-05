# pandit ☸️ pandas utils 
Pandas with some cool additional features

### Installation and usage
`pip install pandit`
```python
import pandas as pd, import pandit
# or import pandit as pd

df=pd.read_tsv(path)
df.sieve(x=3).show()
#Pandas behaves normally otherwise
```
##### If credentials are needed:
```python
import credentials # you manage that part
assert credentials.sheets # credential dict in https://docs.gspread.org/en/latest/oauth2.html
assert credentials.dropbox # your dropbox API key
pd.credentials = credentials
```

### `sieve`
```python
df.sieve(column1=value1, columns2=value2)
# returns df rows where column equals value - if value is not a list, otherwise:
df.sieve(column3=[value1,value2])
# returns df rows where column is value1 or value2; use [[value1,value2]] to match lists
# It's like pd.query but with a pythonic syntax instead of the sql string.
```

### `show`
```python
df.show() # shows multiple rows column by column (one line per column) with nice formatting, one line per column
# ideal for inspecting NLP datasets
df.rshow(n) # random sample of size n (default is 20)
```

Also:

### `df.bold_max()`
bold max float values `df.bold_max().to_latex()`
### `pd.read_tsv`
`read_csv` with `sep='\t'` for lazy persons
### `pd.read_jsonl`
### `pd.read`
`df.read_{extension}` where extension is extracted from the input path (.csv = read_csv)
### `pd.read_wandb(project_name)`
### `df.drop_constant_column`
drop columns that are constant
### `df.to_dropbox(path, format=None, token=None,**kwargs)`
Save dataframe to dropbox
### `df.to_sheets(id,sheet_name,credential=None, include_index=False)`
Save dataframe to sheets
###  `df.undersample(column='label',sampling_strategy='auto',random_state=None,replacement=False)`

