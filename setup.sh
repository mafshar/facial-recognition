#!/bin/bash
#
# Sets up virtual env and configures PATH for the project.
# Install dependencies in virtual environment
#


#############################
# System Requirements (!!!) #
#############################

# Reload environment variables (.bash_profile in Mac; .profile in Ubuntu)
source ~/.bash_profile 2>/dev/null || source ~/.profile

# see if Xcode is downloaded on your machine
xcode-select -p


# updating Homebrew
brew update

# Launch the virtual environment if it exists
VIRTUALENV="deepface"
if workon $VIRTUALENV 2>/dev/null; then
	echo "Launching virtual environment '$VIRTUALENV'..."
else
	echo "Creating virtual environment '$VIRTUALENV'..."

  # https://virtualenvwrapper.readthedocs.org/en/latest/
  # - "-p `which python2.7`": gives location of python2.7 in your machine
  # - "no-site-packages": don't borrow any libraries from your machine (total isolation)
  # - "$VIRTUALENV": name of virtual environment
  mkvirtualenv -p `which python2.7` --no-site-packages $VIRTUALENV

  # Install the project's requirements from requirements.txt
  pip install -r requirements.txt
fi

# additional packages and libraries
brew install mysql-connector-c
brew install cmake pkg-config
brew install jpeg libpng libtiff openexr
brew install eigen tbb

###############
# Final works #
###############

# Append project's subdirectories to the path in order to be able to call their
# modules from python later.
# if [ -z "$PROJECT_ROOT" ]; then
#   export PROJECT_ROOT=$PWD/retrieval
# 	echo "Using PROJECT_ROOT=$PROJECT_ROOT"
# fi
#
# echo "Appending project directories to PYTHONPATH..."
#
# export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
#
# for dir in $PROJECT_ROOT/*; do
# 	export PYTHONPATH=$PYTHONPATH:$dir
# done
