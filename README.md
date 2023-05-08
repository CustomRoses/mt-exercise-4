# MT Exercise 4: Layer Normalization for Transformer Models

This repo is a collection of scripts showing how to install [JoeyNMT](https://github.com/joeynmt/joeynmt), download
data and train & evaluate models, as well as the necessary data for training your own model

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3.10 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository or your fork thereof in the desired place:

    git clone https://github.com/moritz-steiner/mt-exercise-4

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Make sure to install the exact software versions specified in the the exercise sheet before continuing.

Download Moses for post-processing:

    ./scripts/download_install_packages.sh


Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved. It is also possible to continue training from there later on.
