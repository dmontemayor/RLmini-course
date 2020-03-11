# RLmini-course

![CircleCI](https://img.shields.io/circleci/build/github/dmontemayor/RLmini-course?logo=CircleCI)
![GitHub last commit](https://img.shields.io/github/last-commit/dmontemayor/RLmini-course?logo=github)
![GitHub](https://img.shields.io/github/license/dmontemayor/RLmini-course)

A reinforcement learning mini-course in 4 lectures. RL theory with practicum material in Python. Covers tabular and functional methods, including online deep Q learning, and offline off-policy methods. We will be following Sutton and Barto 2018 some what closely. The text book is licenced under the [Creative Commons Attribution-NonCommercial-NoDerivs 2.0 Generic License](http://creativecommons.org/licenses/by-nc-nd/2.0/). You can find a copy of the text book in the `docs` folder.

#Lecture Topics
1. Bandits, Action Value Methods, and Non-Stationary Targets
2. Dynamic Programming, Monte Carlo, and Temporal Difference Learning
3. Artifical Neural Networks
4. Put it all together for Deep Reinforcement Learning

I encourage you to be prepared to discuss specific datasets/research questions
you may be facing so that we can think about how to best cast as an RL implementation. 


# Getting Started
It is recomended you work in some kind of self contained environment so your
python settings don't get messed with. There are a few options such as
[venv](https://docs.python.org/3/library/venv.html)
or [conda](https://docs.conda.io/projects/conda/en/latest/) which we will walk you through in a bit.

# Install with Makefile (Recomended)
Before you go any further, make sure you have Python and that it’s available
from your command line. You can check this by simply running:
```
python --version
```
Ideally, you should get some output like 3.x.x. If not you may have python3 but have to call it by name.
```
python3 --version
```
If it turns out you do not have Python at all, please install
the latest 3.x version from [python.org](python.org).
If you installed Python from source, with an installer from
[python.org](python.org), or via [Homebrew](https://brew.sh/) you should already
have pip. Check if you have pip.
```
pip --version
```
If you’re running Linux and installed using your OS package manager, you may have to install pip separately with commands like
```
apt install python3-pip	# for python 3 (recommended)
                        # or
apt install python-pip	# for python 2
```

A Makefile is provided to make building, installation, and running the code easy. It uses [venv](https://docs.python.org/3/library/venv.html) to create a virtual environment for you.
If you (skipped ahead and manually) activated your virtual environment go ahead and deactivate it.
This Makefile will only rebuild the virtual environment if changes are made to
the requirements file, which is nice.
Again, make sure the virtual environment is deactivated and install with the command.
```
make
```
Throughout the course we'll be introducing example code.
You are welcome to modify the Makefile to run these examples.
For now, I provided a simple example code in `rlmini/example.py` that can
be run with the command
```
make example
```
Devs, you can test the code with the command.
```
make test
```
I trust you can take it from there.

## I hate Makefiles and want to manually setup my virtual environment (Not Recomended!)

### (Option 1) Setup a virtual environment with venv

[venv](https://docs.python.org/3/library/venv.html) is a tool to create isolated
Python environments. venv creates a folder which contains all the
necessary executables to use the packages that a Python project would need.

1. Create a virtual environment named 'venv' for this project in this
project's directory:
```
cd path/to/this/project
python3 -m venv venv
```
2. For MacOS & Linux - Activate the virtual environment to begin using it:
```
source venv/bin/activate
```
The name of the current virtual environment will now appear on the left of the
prompt (e.g. `(venv)Username@Your-Computer project_folder %`) to let you know that
it’s active.

For Windows, the same command mentioned in step 1 can be used to create a
virtual environment. However, activating the environment requires a slightly
different command. Assuming that you are in your project directory the command is.
```
C:\Users\SomeUser\project_folder> venv\Scripts\activate
```
3. Everyone - To deactivate the virtual environment use the command.
```
deactivate
```

### (Option 2) Setup a virtual environment with conda
The conda package and environment manager is included in all versions of
[Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary),
[Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary),
and [Anaconda Repository](https://docs.continuum.io/anaconda-repository/).
Here we will go with the lightweight miniconda option.
Follow the links to get directions on downloading the miniconda installer for
[Windows](https://conda.io/docs/user-guide/install/windows.html),
[MacOS](https://conda.io/docs/user-guide/install/macos.html), or
[Linux](https://conda.io/docs/user-guide/install/linux.html) and run the
installer appropriate for your operating system. For example, the MacOS
directions say the following (**please follow the directions that apply to you**):
+ Open a terminal and run
```
bash Miniconda3-latest-MacOSX-x86_64.sh
```
which will open an installer screen that will walk you through installation.
+ Double check conda is updated.
```
conda update -n base conda
```
+ Create a virtual environment named 'venv' (N.B. Unlike with virtualenv, conda
environments are stored in the same place so ideally unique projects should
have uniquely named virtual environments. We use the name 'venv' here but
you should use a unique name.)
```
conda create --file requirements.txt -c conda-forge -n venv
```
When prompted new packages will be installed, procceed by pressing `y`.
+ Activate the virtual environment
```
conda activate venv
```
+ Update the virtual environment (when necessary)
```
conda env update -n venv -f requirements.txt
```
+ To deactivate the environment, use
```
conda deactivate
```
+ To remove the environment, use
```
conda env remove -n venv
```
# Enjoy the course!
