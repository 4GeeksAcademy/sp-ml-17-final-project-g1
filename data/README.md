Links to official info of the original Datasets:
    [Crowdflower](https://huggingface.co/datasets/tasksource/crowdflower/blob/main/crowdflower.py)
    [Emotions Dataset](https://github.com/dair-ai/emotion_dataset)
    [Go Emotions](https://github.com/google-research/google-research/tree/master/goemotions)
    [ISEAR](https://github.com/paperswithcode/paperswithcode-data?tab=readme-ov-file)
    [MELD](https://github.com/declare-lab/MELD)
    [SemEval-2018](https://competitions.codalab.org/competitions/17751)
    [Kushagra](https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset)
    For the [RedditPost] dataset, what we have done is scrape more than 10,000 post entries from Reddit. Using the Reddit API, we accessed the most popular subreddits related to emotions, such as r/happy or r/sad, and we kept the titles of the most popular posts, which are usually a summary of what they are going to tell. We then grouped them according to the emotion they convey into a dataset.

    In the folder processed you can find the final Dataset we used. It contains data from all the mentioned Dataset but it's also processed to have only one emotion for each text and this emotion is one of the only five we use in out project. If you want also the processed and vectorized Dataset you will have to run the notebook named data-processing.ipynb from the notebooks folder.


-