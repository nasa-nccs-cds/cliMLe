## cliMLe
Machine Learning workflow development for climate applications

### Requirements
cliMLe uses the Keras framework.

#### Keras Installation instructions 
##### For Anaconda under Linux:

* Start with a brand new shell (no conda env activated):
 ```
   $ conda create -n tensorflow
   $ source activate tensorflow
   $ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl
 ```
* Test tensorflow :
 ```
   $ python
   >>> import tensorflow
   >>>
 ```
* Install Keras:
 ```
   $ pip install numpy scipy
   $ pip install scikit-learn
   $ pip install pillow
   $ pip install h5py
   $ pip install keras
 ```