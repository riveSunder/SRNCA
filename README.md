# SRNCA
**S**ymbolic **R**egression **N**eural **C**ellular **A**utomata
### This is a research library for symbolic regression with inductive biases from cellular automata.

## To set it up:

Clone the repository:

```
git clone https://github.com/riveSunder/SRNCA.git
```

## Create a virtual environment:
Use your environment manager of choice to set up a virtual environment for the project (we've tested setup with venv on MacOS and virtualenv on Ubuntu 18).

```
virtualenv my_env --python=python3
```

or:

```
python3 -m venv my_env
```

Install the dependencies using the requirements text file:

```
pip install -r requirements.txt
```

then install the local code:

```
pip install -e .
```

Then test to see if it works:

```
python -m testing.test_all
```

From this, you should be able to run the tutorials notebooks in the `notebooks` folder.
