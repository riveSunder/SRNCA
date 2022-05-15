# SRNCA

**S**ymbolic **R**egression **N**eural **C**ellular **A**utomata

### This is a research library for symbolic regression with inductive biases from cellular automata.

## To set it up:

Clone the repository:

```
git clone https://github.com/riveSunder/SRNCA.git
```

## Create a virtual environment:
Use your environment manager of choice to set up a virtual environment for the project. Replace these steps with those pertaining to your virtual environment manager of choice if needed.

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

install the local code:

```
pip install -e .
```

Run tests:

```
python -m testing.test_all
```

If the tests pass, you can get started with the tutorial notebooks in the `notebooks` folder.


See [coverage.txt](coverage.txt) for the latest `test_commit` coverage.
