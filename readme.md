# Deep Learning Specialization by DeepLearning.AI

# Usage:
1. clone repository with `git clone https://github.com/08Miguel24/deep-learning-specialization-hw.git`
2. navigate to directory with `readme.md` and `requirements.txt` file
3. run command; `conda create -n <name of env e.g. deep-learning-specialization-hw> python=3.10.9`. Note that 3.10.9 must be the python version otherwise packages to be installed would not be compatible with a different python version
4. once environment is created activate it by running command `conda activate`
5. then run `conda activate deep-learning-specialization-hw`
6. check if pip is installed by running `conda list -e` and checking list
7. if it is there then move to step 8, if not then install `pip` by typing `conda install pip`
8. if `pip` exists or install is done run `pip install -r requirements.txt` in the directory you are currently in

# To Do:
add the ff to NeuralNetwork class:
- <s> add regularizer to improve accuracy </s>
- add mean normalizer 
- <s> fix confusing arguments of methods in ShallowNetwork class </s>
- build getter and setter of params
- add dropout
- <s> add visualizer of cost </s>
- add gradient checker
- add experimentation of different hyper parameters
- add predictor function
- <s> measure accuracy and its precision </s>

# Side Notes:
1. The coursera hub is the workspace which contains the notebook, helper files, data sets, and images. To go to the hub, you should first be in the notebook: 
2. Click on "file" and this will lead you to an environment that has all your programming exercises and datasets. You should go there to check out any helper functions that we have provided for you. 
3. To submit the assignment, click on the blue button in the above image labelled "Submit Assignment."
4. Sometimes if the notebook blocks or if you want to clear all the variables and start all over again, rather than quitting the notebook  and opening it again, you could restart the kernel and clear the output if you accidentally end up in some sort of infinite loop
5. to save progress follow click path File > Save and Checkpoint
6. if jupyter hangs or does not respond follow click path Kernel > Restart



HOW TO DOWNLOAD YOUR NOTEBOOK
Open your assignment/lecture notebook.

Go to top left, “File → Open…”

When the page opens, select the checkbox next to your assignment/lecture notebook name and then click “Shutdown”.

When it shutdowns, select the checkbox next to your assignment/lecture notebook name and then click “Download”.



HOW TO REFRESH YOUR WORKSPACE

This will come in handy whenever you'd need to fetch the (latest) assignment and/or other files. And in cases of opening the assignment runs into a 404 error.

1. open the assignment
2. After the assignment opens up, click "File" (top left) and then "Open..."
3. When your workspace opens, select the check box before your assignment file. After it is selected, press "Shutdown".
4. Using the same procedure mentioned above, "Rename" your file. For instance you can change it from assignment_name.ipynb to assignment_name_v2.ipynb. By doing this you'll be able to save your current progress on the assignment after the latest assignment file is fetched.
5. Using the same procedure, "Delete" any other file, if any, that you want to get a fresh copy of.
6. After renaming your file, click on the "Help" button on the top right of the page. From the panel that opens, click "Get latest version" button, and then "Update lab".
7. After the page reloads go to File --> Open... , as described in (2)
8. Now you'll see two notebook files. The one you renamed (as done in step 4) will have your previous progress, and the latest version. 
(Depending on how many previous versions you have kept saved, you could have more than two assignment files in the worksapce)

Option 2 - via URL
1. Follow the steps 1 to 5 as mentioned in Option 1.
2. In the URL of the page append "?forceRefresh=true" at the end of it. For instance, coursera.org will become coursera.org?forceRefresh=true
3. Follow the steps 7 and 8.





