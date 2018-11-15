# DeepMind Health Research Technical Exercise

## Aim of this exercise
Our emphasis here is to appreciate your technical skills, especially your ability to:
* quickly understand unfamiliar ideas and problems
* work with unfamiliar technical concepts
* find and use existing code efficiently
* understand limitations and benefits of the approach you take
* *implement the algorithm and get it to work*
* *apply and briefly explain it*

All these skills are very relevant to the daily work of research engineers as well as research scientists.

Our main interest is in how quickly you can understand the material and get it working. We are not expecting a splendid piece of software engineering, but some efficient and clear code that is a healthy basis for your further research work.

## Possible context
We are interested in trying out a new idea for 3D medical imaging segmentation. We have identified a small, open source dataset to start testing our idea on. As with most research ideas, we want to start with writing a simple baseline against which we can compare our new idea.

## Timeframe
This should take you less than 2 days of your time, depending on your experience, roughly a weekend’s worth of work. We do not expect nor ask you to spend any longer on it.

## Resources
[Three ZIP files](https://drive.google.com/open?id=0B40x2-JgzvT1eVo4NS1qUHhwWFE): train.zip, leaderboard.zip and test.zip. The files were downloaded directly from the [NCI-ISBI 2013 challenge website](https://wiki.cancerimagingarchive.net/display/DOI/NCI-ISBI+2013+Challenge%3A+Automated+Segmentation+of+Prostate+Structures). For more information about the data, see the URL from the competition. The segmentations are in the directories suffixed with “-segm”.
You can use **any code** that is publicly accessible online and of which the license permits its use in the exercise.

## Language
*Core* TensorFlow + Python. Core TensorFlow means no TF Slim, no Keras or other wrappers.

## Deliverable
A report (maximum three pages without images) together with a Jupyter notebook, as described below, and the **commented** source code on GitHub.

## Tasks:
1) **Main task:** write and train a three-dimensional baseline segmentation model in Python/Core TensorFlow (no wrappers such as e.g. TF Slim or Keras) using a basic [3D U-net network](https://arxiv.org/abs/1606.06650) on the provided dataset (you can use train.zip and leaderboard.zip for training). You are free to make your own choices for some of the hyperparameters but don’t spend too much time on this. To get your baseline score, you will measure the segmentation performance on the independent provided test set (test.zip) for both classes using one of standard metrics for segmentation: (mean) Intersection Over Union or IOU (less commonly known as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)). Be aware that metrics are always calculated on the original resolution.

2) *Jupyter notebook (keep the length reasonable!):* For an example of a notebook, see [here](https://github.com/tensorflow/tensorflow/blob/fc49f43817e363e50df3ff2fd7a4870ace13ea13/tensorflow/examples/tutorials/deepdream/deepdream.ipynb). Demonstrate that
* You have explored and verified the data, as well as your processing of it.
* You can load your trained model and get the segmentation prediction for a random test scan.
* Investigate the results of your trained model on the different datasets. Explore failure cases as well as successful cases.

3) Write a short report, no more than three A4 pages (images excluded), exported as PDF to the repository:
* Concisely describe your baseline model and report its performance in a validated way as you would do for a paper (using the test dataset and IOU described in the main task). Make sure this is done in a rigorous way! For reference, see Table 1, Table 2 and Table 3 in the [3D U-net paper](https://arxiv.org/abs/1606.06650).
* Write one or two paragraphs commenting on the *model*: which choices did you make for the model and why? If you deviated from the basic 3D U-net model, even though we don’t ask or expect this from you, you need to give good quantitative arguments. If you can find prior work on the dataset, how does your model compare?
* Write one or two paragraphs on the software engineering decisions you made for the code: did you look into using any public code? If you did and haven’t used any, why not? Do you think there are some bottlenecks in the code limiting performance? Where did you spend most of your time on? What worked? What didn’t work?
* What other extensions, improvements or algorithms you could investigate next (code and/or model).



## Important notes
* Do not rush; it’s fine if you haven’t finished everything within the allotted time. The report is at least as important as the code. We are more interested in how you approach these problems and why you made certain decisions.
* We know the exercise isn’t perfectly defined, this is intentional. However, where you have to make choices, write them in the report and share your reasoning.
* For this exercise, writing tests is low priority. The code should also be fairly light-weight without too many new methods needing tests.
* We are aware that you might not have much experience with Python and/or TensorFlow yet but there should be enough public resources for you to get started quickly.
* Reference any code or papers you use properly (in the code and report)!
* The code should be the basis for further research. As such, the code should initially be fairly low level (no classes, wrapper functions with many parameters, etc.) and transparent so that you can easily test new variations (e.g., the model should be defined in your code and not just imported).
* Provide some instructions on how to use your code.
* Preferably the code runs on a GPU but if you do not have one, it’s fine to use CPU.
* If your model isn’t fully trained within the allotted time, share your preliminary results and state this in your report.
* If you have any questions, email bridie@google.com

Good Luck!