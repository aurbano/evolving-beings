# Evolving Beings

This project creates an environment similar to OpenAI's Gym to play around with simulated "beings" similar to tiny ants or something like that.

These beings experience hunger and thirst, and will die if they run out of energy (so they need to learn to move towards food and water and eat/drink)

They also have a basic vision system, and eventually will be able to change part of their color as a potential form of communication.

------

The main objective of this is to explore intelligent emergent behavior - perhaps given enough time they will learn to communicate and collaborate towards a common goal.

Eventually every parameter in their brains will be "evolvable", allowing certain beings to experience less hunger/thirst for instance, changing their vision capabilities...

![Main UI](screenshots/main.png?raw=true "Main UI of evolving beings")

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

