# chatbot
Third year AI project for an introductory course at Ontario Tech. This project uses the popular Anaconda distribution of Python.

# Setup
### For Windows 10
Download the Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019. <br />
https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0 <br />

Download and install the latest CUDA Toolkit <br />
https://developer.nvidia.com/cuda-downloads <br />

Windows Anaconda <br />
https://www.anaconda.com/products/individual#windows <br />

### For Ubuntu 18.04
https://releases.ubuntu.com/18.04/ <br />
https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart <br />

### Next,
Install the latest distribution of Anaconda making note of its installation location and add it to your PATH variable. Once installed, navigate to the chatbot root directory and execute these commands for the chatbot project:<br />

`conda create -n chatbot python=3.6` <br />
`conda activate chatbot` OR `activate chatbot` <br />
`pip install -r requirements.txt` <br />
`conda deactivate` to deactivate the environment. <br />

To run the initial chatbot prototype, run this in the root directory: <br />
`conda activate chatbot` OR `activate chatbot` <br />
`python main.py` <br />

For the build based on Neural Machine Translation (NMT) for the training model in combination with an improved prototype, continue onwards. <br />
This project builds off the RC_2015-01.bz2 file which contains one month's worth of reddit comments (January 2015). To run this project, this file needs to be downloaded and extracted to `chatbot/raw_data`. While in the root directory, use these next commands to prepare and process the data for training: <br /><br />
`python alanbot_db.py` Builds the database .db file. <br />
`python prep_training_data.py` process the data in the database and outputs .from and .to files. <br /> <br />
Take the `train.from` and `train.to` files in the root directory and paste them into the `chatbot/new_data/` directory, replacing the existing sample data with this project's generated data. <br /> <br />
Change directories to the `chatbot/setup/` folder. <br />
`python prepare_data.py` <br />
`cd ../` <br />
`python train.py` <br /><br />

When satisfied with the training, run the `interacte.py` file in the root directory to interact with alanbot.

# Course Project Deliverables:
* Proposal (10%) (2 pages max) (Due on Feb. 7th 11:59pm): Define the input-output behavior of
the system and the scope of the project. What is your evaluation metric for success? Collect
some preliminary data. Note that you can still adjust your project topic within one week after
submitting the proposal.
* Final report (60%) (10-20 pages) (Due on March. 30th 11:59pm): You should have completed
everything (task definition, infrastructure, approach, literature review, coding, and error
analysis). A readme file explain in a step by step how to execute your code with full commands.
You need to use Python.
* Presenting the subject to the class on March 30th, 2021 during the class and tutorial times (more
than one session).

# Course Project Evaluation:
* Task definition: is the task precisely defined and is the motivation clear? does the world
somehow become a better place if your project were successful?
* Approach: did you describe the used methods clearly, with good justification, and testing?
* Data and experiments: have you explained the data clearly, performed systematic experiments,
and reported concrete results?
* Analysis: did you interpret the results and try to explain why things worked (or didn't work) the
way they did? Do you show concrete examples?
* Extra credit: does the project present interesting and novel ideas (i.e., would this be publishable
at a good conference)?

# Resources
Tech with Tim: https://www.youtube.com/watch?v=wypVcNIH6D4&list=PLzMcBGfZo4-ndH9FoC4YWHGXG5RZekt-Q <br />
Sentdex: https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/ <br />
Reddit Dataset Thread: https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/ <br />
