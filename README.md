# facial recognition
Based on the Facebook DNN, DeepFace

NOTE:

Male class ends at 212 with Neal McDonough as the last true label. Nick Frost (not trained on) is label 213.

Female class ends at 108 with Shannen Doherty as the last true label. Portia Doubleday (not trained on) is label 109.

To run:

make sure you have virtual env and virtual env wrapper; if not, type:

```bash
pip install virtualenv virtualenvwrapper
```

and then put these two lines in your `.bashrc` or `.bash_profile`:

```bash
# Virtualenv/VirtualenvWrapper
source /usr/local/bin/virtualenvwrapper.sh
```

clone the repo in your working directory and type:

```bash
. setup.sh
```

NOTE: if python starts to make your terminal session act strangely, type this command in terminal (assuming you have macports installed):
```bash
sudo port selfupdate && sudo port clean python27 && sudo port install python27 +readline
```
