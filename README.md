# The Selective Camera

## Brief Introduction
This program automatically determines whether or not a face is considered to be
"attractive" or not. Using a Neural Network, the face information is parsed as
black-and-white and then given to the parameters which determines whether or
not the score passes the threshold or not.

## Inspiration
I was reading Doraemon (it's a Japanese magna), and there is one gadget called
"The Selective Camera" which only takes pictures of people the camera likes.
While this program will never develop a full sense of awareness like the camera,
it does its best to mimic the thought process of a human brain through neurons
represented by the Neural Network.

## Instructions
Numpy Scipy, and Pillow PIL is needed. If you don't have these modules installed, run

```pip install numpy scipy pillow --user```

(Note: For Linux/Mac with both Python2/3, replace `pip` with `pip3`.
This script uses Python 3 and not 2).

_Note: If you use Anaconda, I don't think you need anything else. 
IIRC Anaconda bundles all of these modules on installation._

For training, fill train_label.json with file name and attractiveness 
(1 for being attractive, 0 otherwise). Fill the training faces into <project_dir>/data/train_data.
and faces for hypothesis testing to <project_dir>/data/test_data/.

Make sure the directory <project_dir>/data/trained_params exists, 
or else Python may throw an error during hypothesis. 

## Notes
I could not include actual training data (faces) here due to privacy reasons.

## Special Thanks
Andrew Ng, for his awesome Coursera Machine Learning course which the concepts
taught led to the creation of this project.
Fujiko F. Fujio, for his creativity, writing Doraemon magna which inspired
this project.

## License.
See LICENSE.txt file. Basically, do what you want as long as you credit me as the
original creator of this program and acknowledge that I am not responsible
for anything bad that happens. (It shoudln't)
