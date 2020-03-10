# RLmini-course
A reinforcement learning mini-course in 4 lectures. RL theory with practicum material in Python. Covers tabular and functional methods, including online deep Q learning, and offline off-policy methods.

# Getting Started
It is recomended you install in some kind of self contained environment so your
python settings don't get messed with. There are a few options such as
[virtualenv](http://pypi.org/project/virtualenv)
or [conda](https://docs.conda.io/projects/conda/en/latest/) which we will walk
you through now.

## (Option 1) Setup a virtual environment with virtualenv
Before you go any further, make sure you have Python and that it’s available
from your command line. You can check this by simply running:
```
python --version
```
You should get some output like 3.6.2. If not you may have python3 but have to call it by name.
```
python3 --version
```
If it turns out you do not have Python, please install
the latest 3.x version from [python.org](python.org).
If you installed Python from source, with an installer from
[python.org](python.org), or via [Homebrew](https://brew.sh/) you should already
have pip. If you’re on Linux and installed using your OS package manager, you may have to install pip separately.

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

## (Option 2) Setup a virtual environment with conda
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
directions say:
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


# Install with Makefile
A Makefile is provided to make building, installation, and running the code easy.
If you activated your virtual environment go ahead and deactivate it.
This Makefile will only rebuild the virtual environment if changes are made to
the requirements file.
Make sure the virtual environment is deactivated and install with the command.
```
make
```
To run a specific example use the command.
```
make example
```
Devs can test the code with the command
```
make test
```
