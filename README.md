# pandit
Pandas with some cool additional features

### `sieve`
```python
df.sieve(column=value) # returns df rows where column equals value
df.sieve(column=[value1,value2]) # returns df rows where column is value1 or value2
```

### `show`
```python
df.show() # shows multiple rows column by column (one line per column) with nice formatting
```

Also:

### `read_tsv`
### `read_jsonl`
### `read_wandb(project_name)`
### `drop_constant`
