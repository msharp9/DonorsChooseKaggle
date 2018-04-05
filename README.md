This is my code for the DonorsChoose Kaggle competition.  This is playground competition, but is interesting since there are prizes and appears to be for charity.  The data is all in csv files and contain teacher's essays you will use to predict if a project will be accepted.  I chose this dataset to explore text data a little bit more.

Created first submission with score of 0.69294 which got me rank 200/314. (0.82191/0.49489)  Not bad, strongly on the correct side of the coin toss.  Many improvements I can still make to my code base, but quite happy since I rarely work in text analysis.

Update:
Found error in my preprocess script in the text analysis.  Data wasn't being added to csv correctly.  None of the text columns were showing as significant in analysis. Correcting this boosted results by 0.1.  Also experimenting with kfolds, boosted trees, different C values in regression model, ect.

Update:
Created a NN model as my next choice.  Decided to use tflearn to gain more experience with it, as it is pretty simple to set up.  Looks like there isn't a way to predict probabilities yet with this frame work.  Will need to fall back to keras.  Also created a lighter preprocessed dataset with less features for faster learning.
