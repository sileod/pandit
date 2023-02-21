# pandit ☸️ 
Pandas with some cool additional features

### Installation and usage
`pip install pandit`
```python
import pandas as pd, import pandit
# or 
import pandit as pd

df=pd.read_tsv(path)
df.sieve(x=3).show()
```

### `sieve`
```python
df.sieve(column1=value1, columns2=value2)
# returns df rows where column equals value - if value is not a list, otherwise:
df.sieve(column=[value1,value2])
# returns df rows where column is value1 or value2; use [[value1,value2]] to match lists
# It's like pd.query but with a pythonic syntax instead of the sql string.
```

### `show`
```python
df.show() # shows multiple rows column by column (one line per column) with nice formatting
df.rshow(n) # random sample of size n (default is 20)
```

Also:

### `bold_max`
bold max float values `df.bold_max().to_latex()`
### `read_tsv`
`read_csv` with `sep='\t'` for lazy persons
### `read_jsonl`
### `read_wandb(project_name)`
### `drop_constant`
drop columns that are constant
