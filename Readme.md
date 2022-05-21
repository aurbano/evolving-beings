# Evolving Beings

![Main UI](https://raw.githubusercontent.com/aurbano/evolving-beings/main/screenshots/main.png "Main UI of evolving beings")

## Running locally

Just run the `run.py` file at the root. It should open up a window titled "Evolving beings" where the world should start animating.

## Development

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

