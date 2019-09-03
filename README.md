# Spirit_Label_CBIR

A content base image retrieval for spirits. Similar to Bibarra by Matteo Ronchetti. In this project, we use the ORB algorithm in OpenCv for Python to do a base CBIR. We set the nfeature of the orb to 1500, 1000 more than its default feature, but kept all other parameters the same. We did a brute force match using the query images against all the reference images. We summed the distance of the closest 100 matches and decided that the reference image with the lowest total distance was the one closely related to the query image. 7 out of 11 images were identified correclty. In the next iteration, we will need to deal with blurry images and use other parameters. We can also see how just using the image of the labels instead of the whole bottle compares to using the whole bottle as a reference.

## Getting Started

### Prerequisites

This project was written in Python 3.6 on Ubuntu 18.04 using a Jupyter Notebook. Please install the following libraries before running the script. Use the package manager pip to install:

```
# pandas
pip install pandas

#numpy
pip install numpy

```

Dowload your operating system's version of Open-CV from https://opencv.org/

Please read the comments in run_script.py to understand my thinking behind my choices.

### Running Script
Run python run_script.py in your terminal.

