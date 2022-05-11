# Evolving Beings

## Conda environment

### Create environment

```bash
conda env create -f environment.yml
```

### Update environment after pulling new changes

```bash
conda env update --file environment.yml  --prune
```

### Update env file after dependencies:

```bash
conda env export > environment.yml
```

