# Information Flow Bug Classifier

This repository contains source code & data for deciding if a bug is an info-flow bug given its description.

## Setup

Make sure you have Python 3.9 installed. Then, activate this project's virtual environment by running this command:
```sh
$ source .bin/bin/activate
```

You need to install this project's dependencies in your virtual environment, e.g.:
```sh
$ pip3 install -r requirements.txt
```
Note that you only need to perform this step once.

## Running

```sh
$ python3 main.py
use: main.py [data directory]
```

You can recreate our results by using our data, e.g.
```sh
$ python3 main.py data/
```
