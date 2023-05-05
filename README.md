Download Link: https://assignmentchef.com/product/solved-cse512-hw6-pca-via-successive-deflation
<br>
<h2></h2>

Suppose we have a set of <em>n </em>data points <strong>x</strong><sub>1</sub><em>,…,</em><strong>x</strong><em><sub>n</sub></em>, where each <strong>x</strong><em><sub>i </sub></em>is represented as a <em>d</em>-dimensional column vector. Assume that the data has been centerlized, i.e., having zero mean: . Let <strong>X </strong>= [<strong>x</strong><sub>1</sub>;<em>…</em>;<strong>x</strong><em><sub>n</sub></em>] be the (<em>d </em>× <em>n</em>) matrix where column <em>i </em>is equal to <strong>x</strong><em><sub>i</sub></em>. Define <strong>C</strong>to be the covariance matrix of <strong>X</strong>, where.

Next, order the eigenvectors of <strong>C </strong>by their eigenvalues (largest first), and let <strong>v</strong><sub>1</sub><em>,</em><strong>v</strong><sub>2</sub><em>,…,</em><strong>v</strong><em><sub>k </sub></em>be the first <em>k </em>eigenvectors. These satisfy

<strong>v</strong><sub>1 </sub>is the first principal eigenvector of <strong>C </strong>(the eigenvector with the largest eigenvalue), and as such satisfies <strong>Cv</strong><sub>1 </sub>= <em>λ</em><sub>1</sub><strong>v</strong><sub>1</sub>. Now define <strong>x</strong>˜<em><sub>i </sub></em>as the orthogonal projection of <strong>x</strong><em><sub>i </sub></em>onto the space orthogonal to <strong>v</strong><sub>1</sub>:

Finally, define <strong>X</strong><sup>˜ </sup>= [<strong>x</strong>˜<sub>1</sub>;<em>…</em>;<strong>x</strong>˜<em><sub>n</sub></em>] as the deflated matrix of rank <em>d </em>− 1, which is obtained by removing from the <em>d</em>-dimensional data the component that lies in the direction of the first principal eigenvector:

<strong>X</strong><sup>˜ </sup><strong>X</strong>

<ol>

 <li>[7 points] Show that the covariance of the deflated matrix,</li>

</ol>

<strong>C</strong>˜ <sup>1</sup><strong>X</strong>˜<strong>X</strong>˜ <em>T</em>

is given by

<strong>C</strong>˜ 1<strong>v</strong>1<strong>v</strong>1

<em>(Hint: Some useful facts: </em><em> is symmetric, </em><strong>XX</strong><em><sup>T</sup></em><strong>v</strong><sub>1 </sub>= <em>nλ</em><sub>1</sub><strong>v</strong><sub>1</sub><em>, and </em><strong>v</strong> <em>. Also, for any matrices </em><strong>A </strong><em>and </em><strong>B</strong><em>, </em>(<strong>AB</strong>)<em><sup>T </sup></em>= <strong>B</strong><em><sup>T</sup></em><strong>A</strong><em><sup>T</sup></em><em>.)</em>

<ol start="2">

 <li>[7 points] Show that for <em>j </em>6= 1, if <strong>v</strong><em><sub>j </sub></em>is a principal eigenvector of <strong>C </strong>with corresponding eigenvalue <em>λ<sub>j </sub></em>(that is, <strong>Cv</strong><em><sub>j </sub></em>= <em>λ<sub>j</sub></em><strong>v</strong><em><sub>j</sub></em>), then <strong>v</strong><em><sub>j </sub></em>is also a principal eigenvector of <strong>C</strong><sup>˜ </sup>with the same eigenvalue <em>λ<sub>j</sub></em>.</li>

 <li>[8 points] Let <strong>u </strong>be the first principal eigenvector of <strong>C</strong><sup>˜</sup>. Explain why <strong>u </strong>= <strong>v</strong><sub>2</sub>. (You may assume <strong>u </strong>is unit norm.)</li>

 <li>[8 points] Suppose we have a simple method <em>f </em>for finding the leading eigenvector and eigenvalue of a positive-definite matrix, denoted by [<em>λ,</em><strong>u</strong>] = <em>f</em>(<strong>C</strong>). Write some pseudocode for finding the first <em>k </em>principal basis vectors of <strong>X </strong>that only uses the special <em>f </em>function and simple vector arithmetic.</li>

</ol>

<em>(Hint: This should be a simple iterative routine that takes only a few lines to write. The input is </em><strong>C</strong><em>,k, and the function </em><em>f, the output should be </em><strong>v</strong><em><sub>j </sub>and </em><em>λ<sub>j </sub>for </em><em>j </em>∈ 1<em>,</em>··· <em>,k</em><em>)</em>

<h2>2         Action recognition with CNN (35 points + 10 bonus)</h2>

In this question, you will train a convolutional neural network (CNN) to classify images and videos using Pytorch. We use the UCF101 data (see http://crcv.ucf.edu/data/UCF101.php). There are also 10 classes of data in this homework but the data and the number of classes are different from those of Homework 4. Each clip has 3 frames and each frame is 64 ∗ 64 pixels. The labels of train and validation clips are provided in <em>hw</em>6 <em>data.mat </em>inside the directory Question2.

You will first train a CNN for action classification for each image. You will then improve the network architecture and submit the classification results on the test data to Kaggle. Then, you will train a CNN using 3D convolution for a set of video frames (rather than for individual frames), and submit your results to Kaggle.

The detail instructions and questions are in the jupyter notebook <em>Action CNN.ipynb</em>. In this file, there are 8 ‘ToDos’ spots for you to fill. The score of each ToDo is specified at the spot. For the 5<em><sup>th </sup></em>and 8<em><sup>th </sup></em>ToDos, you need to submit CSV result files to Kaggle. The results would be evaluated by Categorization Accuracy.For the 5<em><sup>th </sup></em>ToDo, submit to https://www.kaggle.com/c/cse512hw6image. For the 8<em><sup>th </sup></em>ToDo, submit to https://www.kaggle.com/c/cse512hw6video.

We will maintain a leader board for each Kaggle competition, and the top three entries at the end of the competition (official assignment due date) will receive 10 bonus points. Any submission that rises to top three after the assignment deadline is not eligible for bonus points. The ranking will be based on the Categorization Accuracy. Marks for these questions will be scaled according to the ranking on the Private Leaderboard. To prevent exploiting test data, you are allowed to make a maximum of 2 submissions per 24 hours. Your submission will be evaluated immediately and the leader board will be updated.

Environment setting

Please make a <em>./data </em>folder under the same directory with the <em>Action CNN.ipynb </em>file. Put data <em>trainClips</em>, <em>valClips</em>, <em>testClips </em>and <em>hw</em>6 <em>data.mat </em>from the <em>Question</em>2 directory under <em>./data</em>.

We recommend using virtual environment for the project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run the following in the command-line interface:

cd your_hw6_folder

<table width="0">

 <tbody>

  <tr>

   <td width="288">sudo pip install virtualenv</td>

   <td width="297"># This may already be installed</td>

  </tr>

  <tr>

   <td width="288">virtualenv .env</td>

   <td width="297"># Create a virtual environment</td>

  </tr>

  <tr>

   <td width="288">source .env/bin/activate</td>

   <td width="297"># Activate the virtual environment</td>

  </tr>

 </tbody>

</table>

pip install -r requirements.txt # Install dependencies # Note that this does NOT install TensorFlow or PyTorch, # which you need to do yourself.

# Work (hard) on the assignment # … and when you’re done:

deactivate                                                                                         # Exit the virtual environment

Note that every time you want to work on the assignment, you should run ‘source .env/bin/activate’ (from within your hw6 folder) to re-activate the virtual environment, and deactivate again whenever you are done.

<h2>3         Action Classification Using RNN</h2>

In this section, you will train recurrent neural networks (RNNs) to classify human actions. RNNs are designed handle sequential data.

For human action recognition, you will be using skeleton data that encodes the 3D locations of 25 body joints. The data is collected by Kinect v2. There are 10 different action classes. There are 4000 training sequences, 800 validation sequences, and 1000 test sequences. Each sequence has 15 frames, each frame is a 75-dimension vector (the xyz positions of 25 joints). Data and the Jupyter notebook for this question are provided inside the directory Question3.

You will first train an LSTM for action classification. Then try to improve the network architecture and attach your results with the jupyter notebook. Also add the hyper-parameters explored.

The detailed instructions and questions are in the jupyter notebook <em>RNN ActionClassify.ipynb</em>. In this file, there are 4 ToDo locations for you to fill. The score of each ToDo is specified at the spot. You will need to install the following extra packages: