## cliMLe
Machine Learning workflow development for climate applications

### Requirements
cliMLe uses the Keras framework.

#### Keras Installation instructions 
##### For Anaconda:

* Start with a brand new shell (no conda env activated):
 ```
   $ conda create -n tensorflow -c conda-forge -c cdat cdat
   $ source activate tensorflow
 ```
* Linux Install
 ```
   $ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl
 ```
* OSX Install
 ```
   $ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl 
 ```
 
* Test tensorflow :
 ```
   $ python
   >>> import tensorflow
   >>>
 ```
* Install Keras:
 ```
   $ pip install pillow h5py scikit-learn keras
 ```
 
 * Install additional packages:
 ```
   $ conda install xarray
 ```