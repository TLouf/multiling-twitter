# Multilingualism study with Twitter data

First tests on the multilingualism project on previously acquired data.

## Project Organization
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── .env (x)           <- File containing environment variables loaded with dotenv
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── requirements_geo.txt   <- The requirements file for geographical packages, which may require
|                             prior manual steps
├── data (x)
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modelling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
|
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to pre-process data
│   │
│   ├── utils          <- Utility scripts
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
|
└── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
```

(x) means they're excluded from version control.

`src` module inter-dependencies (generated with [`pydeps`](https://github.com/thebjorn/pydeps)):

![alt text](../master/references/src_deps.svg?raw=true&sanitize=true)


To avoid sharing private data, like the contents of tweets for instance, we
filter out the notebooks' outputs by adding a `.gitattributes` file in
`notebooks/`, which calls a filter defined in `.git/config` by the following script:

```
[filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```


## Install instructions
Install all "classic" dependencies with

```
pip install -r requirements.txt
```

To install `pycld3`, you'll need to follow the instructions from there:
https://github.com/bsolomon1124/pycld3. Windows doesn't seem to be supported
for now.

Then for `geopandas` and its dependencies, it depends on your platform.

#### Linux
The problem here is to install `rtree`, and in particular its C dependency `libspatialindex`. There are three solutions to this.

- The first is simple, but makes you use `conda`, so your whole environment then needs to be built with `conda`:


```
conda install --file requirements_geo.txt
```


- The second solution just takes one more command, but installs `rtree` system-wide. You simply do

```
sudo apt-get install python3-rtree
pip3 install -r requirements_geo.txt
```


- The third is the most flexible, as it allows to install `rtree` in your
  environment, and  to install `libspatialindex` without root privileges.
  You first install `libspatialindex` in your local user directory

```
curl -L http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz | tar xz
cd spatialindex-src-1.8.5
./configure --prefix=/home/<username>/<dir>
make
make install
```

You then add
`SPATIALINDEX_C_LIBRARY=/home/<username>/<dir>/lib/libspatialindex_c.so` as an
environment variable (in `.profile` for instance), and then in your virtual
environment you can just

```
pip3 install -r requirements_geo.txt
```


####  Windows
Download the wheels of `GDAL`, `Rtree`, `Fiona` and `Shapely` from
https://www.lfd.uci.edu/~gohlke/pythonlibs/ (only the win32 versions work).
Install them manually with pip

```
pip install <path to the .whl file>
```

in this order:
1. GDAL
2. Rtree
3. Fiona
4. Shapely

Then `pip install geopandas` should work!




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
