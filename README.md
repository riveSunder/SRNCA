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
virtualenv my_env --python=python3.8
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

To assess testing line coverage:

```
coverage run -m testing.test_all && coverage report
```

Which should yield a report similar to:

```
Ran 9 tests in 57.792s

OK
Name                          Stmts   Miss  Cover
-------------------------------------------------
srnca/__init__.py                 0      0   100%
srnca/nca.py                    122      0   100%
srnca/utils.py                   71      0   100%
testing/__init__.py               0      0   100%
testing/srnca/test_nca.py        71      0   100%
testing/srnca/test_utils.py      74      0   100%
testing/test_all.py               9      0   100%
-------------------------------------------------
TOTAL                           347      0   100%
```

See [coverage.txt](coverage.txt) for the latest `test_commit` coverage.

### See the demonstration notebook at `notebooks/texture_nca.ipynb` to train NCA to generate image textures.

