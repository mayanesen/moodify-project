#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:24:13 2023

@author: mayanesen
"""

# MOODIFY DATA SCIENCE PROJECT
# by Maya Nesen
# July - August 2023

#%% Project Goals

# This project uses the "Emotions Labeled Spotify Songs" dataset that I found on Kaggle to
# explore different patterns of how different songs impact people's mood.
# I am hoping to apply various data science techniques, such as K-means clustering, correlation, Linear 
# Regression, statistical tests, and possibly other  machine learning algorithms to identify any trends.
# PCA may be used as well to see if any of the labels used to describe different qualities
# or aspects of songs are actually similar.

# There are two datasets being used: one has 278k songs and the other has 1200 songs.
# Since I am not using a workstation, I will primiarly use the one with 1200 songs.

# The goal of this project is to both explore the dataset and also practice key data science techniques
# to prepare for any technical interviews and a career in the field!
# Anything that is marked as "review" in the cell title is not being used towards the project goals and more
# for personal practice.


#%% Information: Columns

# 0. index
# 1. track (song name)
# 2. artist
# 3. duration (ms)
# 4. popularity
# 5. uri
# 6. key
# 7. Danceability: Danceability describes how suitable a track is for dancing based on a combination 
        # of musical elements including tempo, rhythm stability, beat strength, and overall regularity. 
        # A value of 0.0 is least danceable and 1.0 is most danceable.
# 8. Energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity 
        # and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal 
        # has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing 
        # to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
# 9. Loudness: the overall loudness of a track in decibels (dB). Loudness values are averaged across the 
        # entire track and are useful for comparing the relative loudness of tracks. Loudness is the quality 
        # of a sound that is the primary psychological correlate of physical strength (amplitude). Values 
        # typically range between -60 and 0 db.
# 10. Speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively 
        # speech-like the recording (e.g. talk show, audiobook, poetry), the closer to 1.0 the attribute value. 
        # Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 
        # 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, 
        # including such cases as rap music. Values below 0.33 most likely represent music and other 
        # non-speech-like tracks.
# 11. Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
        # 1.0 represents high confidence the track is acoustic.
# 12. Instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as 
        # instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the 
        # instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. 
        # Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the 
        # value approaches 1.0.
# 13. Liveness: Detects the presence of an audience in the recording. Higher liveness values 
        # represent an increased probability that the track was performed live. A value above 0.8 provides 
        # a strong likelihood that the track is live.
# 14. Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks 
        # with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low 
        # valence sound more negative (e.g. sad, depressed, angry).
# 15. Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, 
        # the tempo is the speed or pace of a given piece and derives directly from the average beat 
        # duration.
# 16. time signature
# 17. Labels: {'sad': 0, 'happy': 1, 'energetic': 2, 'calm': 3}



#%% Importing all packages I may need
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt  # so we can make figures
from scipy.stats import bootstrap  # to do bootstrap in one line!
from sklearn.model_selection import train_test_split  # train test split
from sklearn.linear_model import LinearRegression  # linear regression easily
# This will allow us to do the PCA efficiently
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # this is the logistic sigmoid function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#%% Import the dataset

# I initially started using the dataset I found on Kaggle but it had 278k songs so 278k rows which was 
# much too large for my computer to handle because I am using a Mac Air and not a workstation. 
# The Moodify project has a smaller dataset which they originally began with that has 1200 songs
# so 1200 rows in the dataset, so I will be using that instead.
'''
dataNP = np.genfromtxt('278k_labelled_uri.csv', delimiter = ',', skip_header=1)
dataNP2 = np.genfromtxt('278k_song_labelled.csv', delimiter = ',')

dataPDold = pd.read_csv('278k_labelled_uri.csv')
dataPD2 = pd.read_csv('278k_song_labelled.csv')
'''
# using numpy
dataNP = np.genfromtxt('1200_song_mapped.csv', delimiter = ',', skip_header=1)

# grabbing the header from the dataset separately
f = open('1200_song_mapped.csv')
header = f.readline().split(',')
f.close()

# using pandas
dataPD = pd.read_csv('1200_song_mapped.csv')

dataPDbig = pd.read_csv('278k_labelled_uri.csv')

#%% Pre-processing

# The dataset is very clean: no NaN values.
check = dataPD.isnull().values.any()
print(check)

#%% EDA: Exploratory Data Analysis

# let's find the proportion of songs that are labeled by mood to see how they are spread out
# labels = 'sad': 0, 'happy': 1, 'energetic': 2, 'calm': 3
# we make a pie chart:
a = dataPD['labels'].value_counts()
plt.pie(a.values, labels = a.index, autopct='%.2f')
plt.title('Pie Chart for 1200 Songs')
plt.show()

# so our dataset is equally proportioned between the four categories. 
# just out of curiosity, for our 278k dataset, it is not as equally proportioned.
o = dataPDbig['labels'].value_counts()
plt.pie(o.values, labels = o.index, autopct='%.2f')
plt.title('Pie Chart for 278000 Songs')
plt.show()


# let's look at some statistics
def descriptive_statistics(matrix):
    median = np.median(matrix)  # median
    sd = np.std(matrix)  # standard deviation
    size = len(matrix)  # size n
    sem = sd / np.sqrt(size)  # standard error
    return np.array([median, sd, size, sem])

# example on "acousticness" feature
acousticness = dataNP[:, 7]
stats_acous = descriptive_statistics(acousticness)
print(stats_acous)


#%% Correlation Matrix
# There are 9 features
features = dataNP[:, 7:16]
corrMatrix = np.corrcoef(features, rowvar= False)
# Plot the correlation matrix: (not that helpful)
plt.imshow(corrMatrix)
plt.xlabel('Features')
plt.ylabel('Features')
plt.colorbar()
plt.show()


# Correlation matrix from Kaggle code by DHRUV CHOUDHARY
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# Select the relevant features
features_label = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
# Calculate the correlation matrix
corr_matrix = dataPD[features_label].corr()
# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Greens', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# 1. Energy and loudness = 0.87 (extremely positive)
# 2. Danceability and valence = 0.54 (moderate)
# 3. Instrumentalness and acousticness = 0.63 (moderate)
# 4. Energy and valence = 0.54 (moderate)
# 5. Loudness and instrumentalness = -0.78 (fairly negative)
# 6. Energy and acousticness = -0.91 (extremely negative)


#%% Linear Regression

# Now, visualizing relationships between features (and to practice linear regression skills)
# I will prioritize here the ones that I noted above as having strong correlations 
# (based on our correlation matrix)


# 1. energy vs acousticness

# independent variable / predictor / x:
energy = dataNP[:, 8].reshape(len(dataNP), 1)
# dependent variable / outcome / y:
acousticness = dataNP[:, 11]
    
# train-test split: prevents any overfitting and makes sure our linear regression model
# is generalizable
x_train, x_test, y_train, y_test = train_test_split(energy, acousticness, test_size=0.5, random_state=42)

x_train = x_train.reshape(600,1)
x_test = x_test.reshape(600,1)
y_train = y_train.reshape(600,1)
y_test = y_test.reshape(600,1)

# build the linear regression model using training data
model = LinearRegression().fit(x_train, y_train)
b1, b0 = model.coef_, model.intercept_

# predicted danceability / outcome using test data
y_pred = model.predict(x_test)
# Or do this:
# y_hat = b1 * x_test + b0

# using RMSE to determine the accuracy of our model
# (the lower the RMSE the better the prediction!)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(rmse)

# can also look at R^2
rSqr = model.score(energy, acousticness)
print(rSqr)

# plot the linear regression
plt.scatter(x_test, y_test, s=4, marker="o", color = 'forestgreen')
#plt.plot(x_test, y_pred, color = 'dimgray')
plt.ylabel('Acousticness')
plt.ylim(-0.1,1)
plt.xlabel('Energy')
plt.suptitle("Energy vs. Acousticness")
plt.title(f'RMSE = {rmse:.3f}', fontsize=10)
plt.show()

# Not sure if linear regression really applies here. Seems more binary.

# just checking Spearman's correlation for this, but we get rho = 0.3469
# which is similar to r = 0.38
from scipy.stats import spearmanr
corr_EnAco, p_value = spearmanr(energy, acousticness)
print("Spearman's correlation coefficient:", corr_EnAco)



# for Energy and Danceability
# the RMSE = 0.157 so we have a fairly good prediction, but still, from the plot, we can see that
# there is a bit of cluster on both extremes of the best fit line, around 0.0-0.2 energy and one
# around 0.8-1.0 energy.



# 2. energy vs loudness

# independent variable / predictor / x:
energy = dataNP[:, 8].reshape(len(dataNP),1)
# dependent variable / outcome / y:
loudness = dataNP[:, 9]
  
# train-test split: prevents any overfitting and makes sure our linear regression model
# is generalizable
x_train2, x_test2, y_train2, y_test2 = train_test_split(energy, loudness, test_size=0.5, random_state=42)

x_train2 = x_train2.reshape(600,1)
x_test2 = x_test2.reshape(600,1)
y_train2 = y_train2.reshape(600,1)
y_test2 = y_test2.reshape(600,1)

# build the linear regression model using training data
model2 = LinearRegression().fit(x_train2, y_train2)

# predicted danceability / outcome using test data
y_pred2 = model2.predict(x_test2)
# Or do this:
# y_hat = b1 * x_test + b0

# using RMSE to determine the accuracy of our model
# (the lower the RMSE the better the prediction!)
rmse = np.sqrt(np.mean((y_pred2 - y_test2)**2))

# plot the linear regression
plt.scatter(x_test2, y_test2, s=4, marker="o", color = 'forestgreen')
plt.ylabel('Loudness')
plt.xlabel('Energy')
plt.suptitle("Energy vs. Loudness")
plt.title(f'RMSE = {rmse:.3f}', fontsize=10)
plt.show()

# Clearly, Linear Regression doesn't really apply here. The relationship seems to be more exponential.
# Let's try finding Spearman's correlation rather than Pearon's to compare.
from scipy.stats import spearmanr
corr_DanEn, p_value = spearmanr(energy, loudness)
print("Spearman's correlation coefficient:", corr_DanEn)
# Pearson's r = 0.87 and Spearman's rho = 0.897 which are pretty similar.


# the RMSE = 4.293 which is low but still too high to be a good prediction, compared to our previous plot.
# from looking at our plot visually, it's clear that the relationship isn't linear.




# 3. energy vs valence

# independent variable / predictor / x:
energy = dataNP[:, 8].reshape(len(dataNP),1)
# dependent variable / outcome / y:
valence = dataNP[:, 14]
  
# train-test split: prevents any overfitting and makes sure our linear regression model
# is generalizable
x_train2, x_test2, y_train2, y_test2 = train_test_split(energy, valence, test_size=0.5, random_state=42)

x_train2 = x_train2.reshape(600,1)
x_test2 = x_test2.reshape(600,1)
y_train2 = y_train2.reshape(600,1)
y_test2 = y_test2.reshape(600,1)

# build the linear regression model using training data
model2 = LinearRegression().fit(x_train2, y_train2)

# predicted danceability / outcome using test data
y_pred2 = model2.predict(x_test2)
# Or do this:
# y_hat = b1 * x_test + b0

# using RMSE to determine the accuracy of our model
# (the lower the RMSE the better the prediction!)
rmse = np.sqrt(np.mean((y_pred2 - y_test2)**2))

# plot the linear regression
plt.scatter(x_test2, y_test2, s=4, marker="o", color = 'forestgreen')
plt.plot(x_test2, y_pred2, color = 'dimgray')
plt.ylabel('Valence')
plt.xlabel('Energy')
plt.suptitle("Energy vs. Valence")
plt.title(f'RMSE = {rmse:.3f}', fontsize=10)
plt.show()

# Clearly, Linear Regression doesn't really apply here. The relationship seems to be more exponential.
# Let's try finding Spearman's correlation rather than Pearon's to compare.
from scipy.stats import spearmanr
corr_EnVal, p_value = spearmanr(energy, valence)
print("Spearman's correlation coefficient:", corr_EnVal)



# 4. instrumentalness vs loudness

# independent variable / predictor / x:
loudness = dataNP[:, 9].reshape(len(dataNP),1)
# dependent variable / outcome / y:
instrumentalness = dataNP[:, 12]
  
# train-test split: prevents any overfitting and makes sure our linear regression model
# is generalizable
x_train2, x_test2, y_train2, y_test2 = train_test_split(loudness, instrumentalness, test_size=0.5, random_state=42)

x_train2 = x_train2.reshape(600,1)
x_test2 = x_test2.reshape(600,1)
y_train2 = y_train2.reshape(600,1)
y_test2 = y_test2.reshape(600,1)

# build the linear regression model using training data
model2 = LinearRegression().fit(x_train2, y_train2)

# predicted danceability / outcome using test data
y_pred2 = model2.predict(x_test2)
# Or do this:
# y_hat = b1 * x_test + b0

# using RMSE to determine the accuracy of our model
# (the lower the RMSE the better the prediction!)
rmse = np.sqrt(np.mean((y_pred2 - y_test2)**2))

# plot the linear regression
plt.scatter(x_test2, y_test2, s=4, marker="o", color = 'forestgreen')
#plt.plot(x_test2, y_pred2, color = 'dimgray')
plt.ylabel('Instrumentalness')
plt.xlabel('Loudness')
plt.suptitle("Loudness vs. Instrumentalness")
plt.title(f'RMSE = {rmse:.3f}', fontsize=10)
plt.show()

# Clearly, Linear Regression doesn't really apply here. The relationship seems to be more exponential.
# Let's try finding Spearman's correlation rather than Pearon's to compare.
from scipy.stats import spearmanr
corr_val, p_value = spearmanr(energy, valence)
print("Spearman's correlation coefficient:", corr_val)


# 5. instrumentalness vs acousticness

# independent variable / predictor / x:
acousticness = dataNP[:, 11].reshape(len(dataNP),1)
# dependent variable / outcome / y:
instrumentalness = dataNP[:, 12]
  
# train-test split: prevents any overfitting and makes sure our linear regression model
# is generalizable
x_train2, x_test2, y_train2, y_test2 = train_test_split(acousticness, instrumentalness, test_size=0.5, random_state=42)

x_train2 = x_train2.reshape(600,1)
x_test2 = x_test2.reshape(600,1)
y_train2 = y_train2.reshape(600,1)
y_test2 = y_test2.reshape(600,1)

# build the linear regression model using training data
model2 = LinearRegression().fit(x_train2, y_train2)

# predicted danceability / outcome using test data
y_pred2 = model2.predict(x_test2)
# Or do this:
# y_hat = b1 * x_test + b0

# using RMSE to determine the accuracy of our model
# (the lower the RMSE the better the prediction!)
rmse = np.sqrt(np.mean((y_pred2 - y_test2)**2))

# plot the linear regression
plt.scatter(x_test2, y_test2, s=4, marker="o", color = 'forestgreen')
#plt.plot(x_test2, y_pred2, color = 'dimgray')
plt.ylabel('Instrumentalness')
plt.xlabel('Acousticness')
plt.suptitle("Acousticness vs. Instrumentalness")
plt.title(f'RMSE = {rmse:.3f}', fontsize=10)
plt.show()

# Clearly, Linear Regression doesn't really apply here. The relationship seems to be more exponential.
# Let's try finding Spearman's correlation rather than Pearon's to compare.
from scipy.stats import spearmanr
corr_val, p_value = spearmanr(acousticness, instrumentalness)
print("Spearman's correlation coefficient:", corr_val)


# 6. valence vs danceability

# independent variable / predictor / x:
valence = dataNP[:, 14].reshape(len(dataNP),1)
# dependent variable / outcome / y:
danceability = dataNP[:, 7]
  
# train-test split: prevents any overfitting and makes sure our linear regression model
# is generalizable
x_train2, x_test2, y_train2, y_test2 = train_test_split(valence, danceability, test_size=0.5, random_state=42)

x_train2 = x_train2.reshape(600,1)
x_test2 = x_test2.reshape(600,1)
y_train2 = y_train2.reshape(600,1)
y_test2 = y_test2.reshape(600,1)

# build the linear regression model using training data
model2 = LinearRegression().fit(x_train2, y_train2)

# predicted danceability / outcome using test data
y_pred2 = model2.predict(x_test2)
# Or do this:
# y_hat = b1 * x_test + b0

# using RMSE to determine the accuracy of our model
# (the lower the RMSE the better the prediction!)
rmse = np.sqrt(np.mean((y_pred2 - y_test2)**2))

# plot the linear regression
plt.scatter(x_test2, y_test2, s=4, marker="o", color = 'forestgreen')
plt.plot(x_test2, y_pred2, color = 'dimgray')
plt.ylabel('Danceability')
plt.xlabel('Valence')
plt.suptitle("Valence vs. Danceability")
plt.title(f'RMSE = {rmse:.3f}', fontsize=10)
plt.show()

# Clearly, Linear Regression doesn't really apply here. The relationship seems to be more exponential.
# Let's try finding Spearman's correlation rather than Pearon's to compare.
from scipy.stats import spearmanr
corr_val, p_value = spearmanr(valence, danceability)
print("Spearman's correlation coefficient:", corr_val)




# 1. Energy and loudness = 0.87 (extremely positive) DONE
# 2. Danceability and valence = 0.54 (moderate)
# 3. Instrumentalness and acousticness = 0.63 (moderate)
# 4. Energy and valence = 0.54 (moderate) DONE
# 5. Loudness and instrumentalness = -0.78 (fairly negative) DONE
# 6. Energy and acousticness = -0.91 (extremely negative) DONE






#%% Review: Partial correlation and Multiple regression
# (doesn't make much sense here but let's just do it anyways for practice)

# Partial correlation: do two simple linear regressions with the confound, then correlating the residuals 

# Let's pretend that energy is a confound (z) between danceability (y) and loudness (x)

# Initialize data for first SLR (OLS): Predicting income from lead levels
z = dataNP[:,9].reshape(len(dataNP),1)  # predictor is energy
x = dataNP[:,12] # outcome is loudness

# 1) Building the model (initialize and fit):
loudModel = LinearRegression().fit(z, x)

# 2) Evaluating the model ("running the model"):
loudSlope = loudModel.coef_ # b1 (slope)
loudIntercept = loudModel.intercept_ # b0 (intercept)
yHatLoud = loudSlope * x + loudIntercept #Predicted loudness from energy

# 3) Determining the residuals 
residuals1 = x - yHatLoud.flatten() #Residuals = Literally the distances between the actual 
# outcomes and the predicted outcomes. Importantly, no squaring. The sign is preserved (!)
# Flatten in the line above to enforce that y and yHat have the same dimensionality, 
# which is important for taking element-wise differences

# Initialize data for second SLR: Predicting danceability from energy
# z = dataNP[:,9].reshape(len(dataNP),1)  # predictor is still energy
y = dataNP[:,8] # outcome is now danceability

# 1) Building (creating and fitting the model):
danceModel = LinearRegression().fit(z, y) #Same syntax, but y means something else now

# 2) Evaluate the model (like above, just with danceability)
danceSlope = danceModel.coef_ # Same as above, but for danceability
danceIntercept = danceModel.intercept_ # Same as above, but for danceability
yHatDance = danceSlope * x + danceIntercept

# 3) Compute residuals:
residuals2 = y - yHatDance.flatten() #This gives us the 2nd residuals

# Last step: Correlate the residuals (what can't be accounted for by the confounds)
partCorr = np.corrcoef(residuals1,residuals2) #This yields the full correlation matrix
print('Partial correlation:',np.round(partCorr[0,1],3))



# Multiple Regression: multiple unknown confounds

# MODELING: ONE PREDICTOR
# convention: put the strongest predictor first; put in one by one
# how do we know which one is strongest? look at the highest/strongest (positive or negative) correlation

# strongest relationship is acousticness and energy: r = -0.91

# we explicitly call the predictors x and the outcome y (by convention)
x = dataNP[:, 11].reshape(len(dataNP), 1) # predictor 1: acousticness
y = dataNP[:, 8] # outcome: energy
singleFactorModel = LinearRegression().fit(x,y) # build the model

# we look at r^2: tells us the amount of the variance in the outcome accounted for 
# in the outcome by this predictor
rSqrSingle = singleFactorModel.score(x,y)
print(rSqrSingle)
# 83.5% of the variance
# pretty good, very strong relationship!
# recall: variance = the difference between predicted values and actual outcomes


# MODELING: TWO PREDICTORS
# we want to put predictors that are as innependent as possible
# the EDA guides our model building:
# if we don't want to build a full model with all our predictors (for reasons discussed in class: overfitting), 
# a smart choice might be to start with the one that has the highest correlation to the outcome and then put 
# the next predictor in that has the lowest correlation to the first predictor (to add independent information)

# so acousticness lowest correlation with = tempo and liveness (r = -0.29)
# let's do liveness
liveness = dataNP[:, 13].reshape(len(dataNP), 1)

# how does r^2 go up as we add predictors? higher correlation with outcome or with predictor? it's a balance
# first predictor vs outcome
# but it also needs to be related to the outcome somewhat. in general, you want to add new information that is 
# related to the outcome of the model

# note the syntax: we use a range of columns (the first two)
X = np.concatenate([x, liveness], axis=1)

twoFactorModel = LinearRegression().fit(X,y)
rSqrTwo = twoFactorModel.score(X,y)
print(rSqrTwo)
# 83.9% of variance in energy explained by liveness and acousticness

# why might you not just put all the predictors in right from the get go? because maybe just two predictors
# would be enough but you would not know if you just do everything at once


# MODELING: ALL PREDICTORS
danceability = dataNP[:, 7].reshape(len(dataNP), 1)
allButEn = dataNP[:, 9:16]
X = np.concatenate([danceability, allButEn], axis = 1)
fullModel = LinearRegression().fit(X,y)
rSqrFull = fullModel.score(X,y)
print(rSqrFull)
# now our model with all predictors accounts for the majority of the variance in outcome (89.9%)
# pretty good!


# let's look at the model
# in addition to just the r^2 (which is interesting), one might now want to know the intercept and 
# slope of the model, especially when controlling for several factors / confounds:
b0t, b1t = fullModel.intercept_, fullModel.coef_

# VISUALIZATION
# we can visualize all the predictors (line, plane, hyperplane, etc. – will it work?)
# but now we need to explicitly evaluate the model by using the coefficients we just computed.
# this is not properly done here (on purpose): we can't reuse the same data we used to build the model to evaluate it too
# that can create overfitting problems. don't do this. we need to split the original data in several ways to avoid this
# but more about that next code session.

yHatt = b1t[0]*dataNP[:,7] + b1t[1]*dataNP[:,9] + b1t[2]*dataNP[:,10] + b1t[3]*dataNP[:,11] + b1t[4]*dataNP[:,12] + b1t[5]*dataNP[:,13] + b1t[6]*dataNP[:,14] + b1t[7]*dataNP[:,15] + b0t 
# Evaluating the model: First coefficient times IQ value + 2nd coefficient * hours worked and so on, plus the intercept (offset)
plt.plot(yHatt,y,'o',markersize=4) # y hat predictors goes on x-axis
plt.xlabel('Prediction from model') 
plt.ylabel('Actual energy')  
plt.suptitle('Multiple Regression: Predicted vs. Actual Energy')
plt.title('R^2 = {:.3f}'.format(rSqrFull), fontsize = 10)
# this is not bad but a bit blobby
# R^2 = 0.899
plt.show()


#%% Review: Ridge and Lasso Regression

from sklearn.linear_model import Ridge #To do ridge regression
from sklearn.metrics import mean_squared_error #To evaluate model with function

alph = 2 # lambda is called ALPHA in Python

'''
#Doing ridge regression on the training set
ridge2 = Ridge(alpha = alph, normalize = True)
ridge2.fit(xTrain, yTrain)                       # Fit ridge regression on  training data
pred2 = ridge2.predict(xTest)                    # Use model to predict test data
print(mean_squared_error(yTest, pred2))          # Calculate the test MSE
'''

from sklearn.linear_model import Lasso #This is to do the LASSO
from sklearn.preprocessing import scale #This is to fit it

'''
numIt = 10000 #How many iterations - Lasso is an iterative algorithm, for convergence

lasso = Lasso(max_iter = numIt, normalize = True) #Create LASSO model
lasso.set_params(alpha=alph) # set hyperparameter lambda ("Alpha" in Python)
lasso.fit(scale(xTrain), yTrain) # fit the model
yetNewBetas = lasso.coef_ # what are our lasso betas
mean_squared_error(yTest, lasso.predict(xTest)) # evaluate the model, calculate the error
'''


#%% Review: Hypothesis Testing / Statistical Tests

# PARAMETRIC TESTS:

'''
# Independent Samples t-test: 2 independent groups
t, p = stats.ttest_ind(group1, group2)
    # t is the t-statistic
    # How likely is such a t-value by chance? That is given by the p-value
    # There is a significant difference. 
    # In english: The difference between the samples (specifically the sample 
    # means) is too large to be reasonably consistent with chance. 

# Paired Samples t-test: 1 group, comparing before/after an effect
t, p = stats.ttest_rel(group_before, group_after)

# Welch t-test: 2 independent groups, but homogeneity of variance is not assumed
tW, pW = stats.ttest_ind(group1, group2, equal_var=0) # turning equal_var off  

# ANOVA: more than 2 groups
f, pA = stats.f_oneway(group1, group2, group3)
    #ANOVAs yield an "f" value as a test statistic
    #The question we have for this f value is what p-value is associated with it
    #The question we have for the p-value is whether it is smaller than the alpha level
    #If it is, we call the difference "statistically significant".
'''   

# However, if a t-test was not an appropriate test and the ANOVA is an extension of the t-test to more 
# than 2 groups (or group means), then the ANOVA was probably also not the right test. 
# We should have used a nonparametric test.


# NON-PARAMETRIC TESTS:

'''
# Chi-Squared test: compare observed and expected frequencies (categorical data)
from scipy.stats import chisquare
chi2_stat, p = chisquare(observed, f_exp = expected)

# Mann-Whitney U test: 2 independent groups (ordinal data, compares medians)
u, pMW = stats.mannwhitneyu(group1, group2)

# Kolmogorov-Smirnov (KS) test: works for more than 2 groups, compares underlying distributions
h, pK = stats.kruskal(group1, group2, group3)
    # Nonparametric test analogous to the ANOVA = "Kruskal-Wallis test"
    # Question for h, our test statistic: How likely is that by chance
    # Question for p, our p-value: Whether it is smaller or larger than alpha
    # If smaller than alpha: Significant
'''

#%% Review: Resampling Methods, Confidence Intervals

'''
# Boostrapping
data = (combinedData[:, whichMovie]) # convert to sequence because that is what scipy expects
bootstrapCI = bootstrap(data, np.mean, n_resamples = 1e4, confidence_level=0.95) # gives us confidence interval

# Permutation Test
def ourTestStatistic(x,y):
    return np.mean(x) - np.mean(y)
dataToUse = (empiricalData1,empiricalData2)
pTest = permutation_test(dataToUse,ourTestStatistic,n_resamples=1e4)    
print('Test statistic:', pTest.statistic)
print('exact p-value:',pTest.pvalue)

# Review: Effect size, Power, Confidence Intervals

'''

#%% PCA: are any of these musical features similar?

# Principal Component Analysis (PCA) is an unsupervised machine learning algorithm.

# PCA: want to find the first principal component: the type of question that can
# encapsulate others well
# going to use a smaller portion of features (first 200 rows/songs)
features = dataNP[:, 7:16]

# 1. Z-score the data:
zscoredData = stats.zscore(features)

# 2. Initialize PCA object and fit to our data:
pca = PCA().fit(zscoredData)

# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_
# If we order the eigenvectors in decreasing order of eigenvalue = "Principal components"

# 3b. "Loadings" (eigenvectors): Weights per factor in terms of the original data.
loadings = pca.components_  # Rows: Eigenvectors. Columns: Where they are pointing
# In other words, not mean centered, not-z-scored data will yield nonsense, if fed to a PCA.

# 3c. Rotated Data: Simply the transformed data - people's ratings (rows) in
# terms of 10 questions variables (columns) ordered by decreasing eigenvalue
# (principal components)
rotatedData = pca.fit_transform(zscoredData)

# 4. For the purposes of this, you can think of eigenvalues in terms of
# variance explained:
varExplained = eigVals/sum(eigVals)*100

# Now let's display this for each factor:
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))


# Scree Plot: bar graph of the sorted Eigenvalues
x = np.linspace(1, 9, 9)
plt.bar(x, eigVals, color='forestgreen')
# Orange Kaiser criterion line for the fox
plt.plot([0, 9], [1, 1], color='dimgray')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('PCA Scree Plot')
plt.show()

# Acousticness is clearly the STRONGEST principal component.
# meaning it describes the other musical features pretty well.

# Acousticness as first principal component: PLOT
# Accounts for 47.568% of the variance
whichPrincipalComponent = 0
# note: eigVecs multiplied by -1 because the direction is arbitrary
plt.bar(x, loadings[whichPrincipalComponent, :]*-1, color = 'forestgreen')
plt.xlabel('Feature')
plt.xticks()
plt.ylabel('Loading')
plt.title("PCA: Acousticness")
plt.show()  # Show bar plot

# from the bar plot, we see that acousticness describes best all the feautres except two,
# except liveness and loudness. this makes some sense since acoustic music is generally not loud
# (think of an acoustic guitar version of a pop song; it would have a softer sound), and 
# is not always performed live, at least for genres of music that aren't heavily using 
# acoustic instruments.

#%% Supervised Machine Learning

# Logistic Regression

'''
x = data[:,0].reshape(len(data),1) 
y = data[:,1]
# Fit model:
model = LogisticRegression().fit(x,y) 
# Plot the model
#Format the data
x1 = np.linspace(260,345,500)
y1 = x1 * model.coef_ + model.intercept_
sigmoid = expit(y1)
# Plot:
plt.plot(x1,sigmoid.ravel(),color='red',linewidth=3) # the ravel function returns a flattened array
plt.scatter(data[:,0],data[:,1],color='black')
plt.hlines(0.5,260,345,colors='gray',linestyles='dotted')
plt.xlabel('GRE score')
plt.xlim([260,345])
plt.ylabel('Admitted?')
plt.yticks(np.array([0,1]))
plt.show()
#Use the fitted model to make predictions:
testScore = 330
probGettingIn = sigmoid[0,np.abs(x1-testScore).argmin()]
print('Probability of getting accepted:',probGettingIn.round(3))
'''


# Decision Trees / Random Forests (Bagging)
'''
# First:
# Mixing in the labels Y (we know the outcomes):
X = np.column_stack((origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]))
plt.plot(X[np.argwhere(yOutcomes==0),0], X[np.argwhere(yOutcomes==0),1],'o',markersize=2,color='green')
plt.plot(X[np.argwhere(yOutcomes==1),0], X[np.argwhere(yOutcomes==1),1],'o',markersize=2,color='blue')
plt.xlabel('Challenges')
plt.ylabel('Support')
plt.legend(['euthymic','depressed'])
plt.show()    
#Note: There is a trend, but the outcomes are not fully determined by these factors - there is variability
# Actually doing the random forest
numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(X,yOutcomes) #bagging numTrees trees
# Use model to make predictions:
predictions = clf.predict(X) 
# Assess model accuracy:
modelAccuracy = accuracy_score(yOutcomes,predictions)
print('Random forest model accuracy:',modelAccuracy)

# We are able to predict ~100% of the outcomes with this model. There are 
# close to no errors. Even the strange cases, we got. The problem is that if 
# you have results that are too good to be true, they probably are not true. 
# We committed the sin of "overfitting", due to the fact that we used the
# same data to both fit ("train") the model and test it.

# The problem is that results from overfit models won't generalize because
# some proportion of the data is due to noise. If you fit perfectly, you fit
# to the noise. The noise will - by definition - not replicate. 

# Prescription: "Don't do that". Use one set of data to build the model and
# another to train it.

# Best solution: Get new data. Rarely practical.
# Most common solution: Split the dataset. There are many ways to do this,
# such as 80/20 (at random). The most powerful - but most computationally
# expensive - is to use leave-one-out: use the entire dataset to build the 
# model, expect for one point. Predict that point from n-1 data. Do that 
# many times - at random - and average the results. 
'''


# KNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

features_mood = dataNP[:, 7:16]
mood = dataNP[:, 17]

# Assuming 'X' contains your feature matrix and 'y' contains your target labels
X_train, X_test, y_train, y_test = train_test_split(features_mood, mood, test_size=0.2, random_state=42)

# Create the KNN classifier (you can specify the number of neighbors 'n_neighbors')
knn_classifier = KNeighborsClassifier(n_neighbors=11)

# Fit the classifier to the training data
knn_classifier.fit(X_train, y_train)

# Predict the classes of test data
y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print()
print("Confusion Matrix:\n", confusion_mat)
print()
print("Classification Report:\n", class_report)





#%% Unsupervised Machine Learning: K-Means Clustering

# An unsupervised machine learning algorithm would not really apply to this dataset since it is labeled.
# Above, I've used supervised machine learning algorithms and PCA to explore my data!

# K-means

# K-means clustering is an unsupervised machine learning algorithm which attempts to find
# an inherent structure or "clustering" in our (unlabeled) data.


x_new = dataNP[:, 7:16] # acousticness and danceability based on PCA (main two principal components)

# SILHOUETTE TO FIND IDEAL k VALUE
# Init:
numClusters = 9  # how many clusters are we looping over? (from 2 to 9)
sSum = np.empty([numClusters, 1])*np.NaN  # init container to store sums

# Compute kMeans for each k:
for ii in range(2, numClusters+2):  # Loop through each cluster (from 2 to 9)
    kMeans = KMeans(n_clusters=int(ii)).fit(
        x_new)  # compute kmeans using scikit
    cId = kMeans.labels_  # vector of cluster IDs that the row belongs to
    # coordinate location for center of each cluster
    cCoords = kMeans.cluster_centers_
    # compute the mean silhouette coefficient of all samples
    s = silhouette_samples(x_new, cId)
    sSum[ii-2] = sum(s)  # take the sum
    # Plot data:
    plt.subplot(3, 3, ii-1)
    plt.hist(s, bins=20)
    plt.xlim(-0.2, 1)
    plt.ylim(0, 20)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    # sum rounded to nearest integer
    plt.title('Sum: {}'.format(int(sSum[ii-2])), fontsize=10)
    plt.tight_layout()  # adjusts subplot
plt.show()

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2, numClusters, 9), sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.title('K-Means: Sum of Silhouette Scores')
plt.show()

# kMeans yields the coordinates centroids of the clusters, given a certain number k
# of clusters. Silhouette yields the number k that yields the most unambiguous clustering
# This number k is the maximum of the summed silhouette scores.

# NOW WE CLUSTER
# Now that we determined the optimal k, we can now ask kMeans to cluster the data for us,
# assuming that k

# kMeans:
numClusters = 3
kMeans = KMeans(n_clusters=numClusters).fit(x_new)
cId = kMeans.labels_
cCoords = kMeans.cluster_centers_

# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x_new[plotIndex, 0], x_new[plotIndex, 1], 'o', markersize=3)
    plt.plot(cCoords[int(ii-1), 0], cCoords[int(ii-1), 1],
             'o', markersize=5, color='black')
    plt.xlabel('X title')
    plt.ylabel('Y title')
    plt.title("TITLE?")





