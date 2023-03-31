# Multilingualism study with Twitter data

[![figshare](https://img.shields.io/badge/doi-10.6084%2Fm9.figshare.20627238.v1-blue.svg?style=flat-square)](https://doi.org/10.6084/m9.figshare.20627238)
[![article](https://img.shields.io/static/v1?label=article&message=10.1103%2FPhysRevResearch.3.043146&color=success&style=flat-square)](https://doi.org/10.1103/PhysRevResearch.3.043146)

This repository contains code I wrote as part of my work on a project studying
multilingualism using Twitter data. It is part of my PhD at the
[IFISC](https://ifisc.uib-csic.es/en/), under the supervision of 
[José Javier Ramasco](https://ifisc.uib-csic.es/users/jramasco/) and 
[David Sanchez](https://ifisc.uib-csic.es//users/dsanchez/). The code is used 
to analyse geo-tagged tweets sent within a set of multilingual countries, which
were acquired over the years by the IFISC' data engineer, Antonia Tugores, using
the [streaming endpoint of the Twitter API](https://developer.twitter.com/en/docs/tweets/sample-realtime/overview/get_statuses_sample).
We attributed one or more languages to users, and a cell of residence, among the
cells we define on a regular grid covering each region of interest. We visualise
and then analyse the distributions of local languages using a set of metrics.
The end goal is to assess the existing models of language competition.

Raw data (which I cannot give access to for privacy reasons) are processed in
`notebooks/1.1.first_whole_analysis.ipynb`, which return the counts of speakers
by language and by cell of residence, available on
[figshare](https://figshare.com/articles/dataset/Spatial_distributions_of_languages_extracted_from_Twitter/14339321).
These can then be analysed in `notebooks/1.2.mixing_metrics_tests.ipynb` and
`notebooks/1.3.EMD.ipynb`.

Instead of delving into details here, I recommend you have a look at the article
published in [Physical Review
Research](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.043146),
also available on [arXiv](https://arxiv.org/abs/2105.02570).


## Project Organization
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── Makefile           <- Makefile to setup the project
├── .env (x)           <- File containing environment variables loaded with dotenv
├── environment.yml    <- The requirements file for reproducing the analysis environment with conda
├── requirements.txt   <- The requirements file for reproducing the analysis environment with pip
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
│   ├── models         <- Scripts to simulate the models
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
`notebooks/`, which calls a filter defined in `.git/config` by the following
script:

```
[filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

## Install instructions
In a terminal, navigate to this directory and simply run `make`.

### conda
If you have `conda` installed, you should have seen 
`>>> Detected conda, creating conda environment.` popping up in your terminal,
if that's the case you should be good to go!

### pip
Otherwise, your environment will be built with `pip` in a directory called
`.venv`. 
All "classic" dependencies  have already been installed with

```
pip install -r requirements.txt
```

To install `pycld3`, you'll need to follow the instructions from there:
https://github.com/bsolomon1124/pycld3. Windows doesn't seem to be supported
for now.

Then for `geopandas` and its dependencies, it's a bit more complicated, and it depends on your platform.

#### Linux
The problem here is to install `rtree`, and in particular its C dependency
`libspatialindex`. There are two solutions to this.

- The first solution just takes one more command, but installs `rtree`
  system-wide. You simply do

```
sudo apt-get install python3-rtree
pip3 install -r requirements_geo.txt
```


- The second is the most flexible, as it allows to install `rtree` in your
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
