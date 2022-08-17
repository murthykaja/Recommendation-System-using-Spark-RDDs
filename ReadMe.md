This is the code for User-User and Item-Item collaborative Recomentation system.

Command: 

spark-submit a3_p2_lastname_id.py 'hdfs:/data/trial_yelp_dataset_review.json'


This code return top 10 users with similar ratings and top 10 items with simlar ratings.
I have also used Cosine similarity for the simialrity score between the items and users rating vectors.
This code uses Spark and the intallly divides the numbers of partitions to be used for fast processing.
