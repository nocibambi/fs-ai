# Basics
import pandas as pd
from matplotlib import pyplot

# Cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading and preparing the data
gtd = pd.read_excel("globalterrorismdb_0617dist.xlsx")

###########################
# Exploratory Data Analysis
###########################

# Basic characteristics of the dataset
gtd.info()
gtd.dtypes.value_counts()
gtd.columns.values

#def get_dtypes(data, datatypes):
#    """The function filters the dataframe for the given datatypes.
#    Parameters:
#            1. data, pandas dataframe
#            2. datatypes, list
#    It returns the original dataframe with the filtered columns.
#
#    Possible datatypes can be listed by the `data.dtypes` command.
#    """
#    filter = []
#    for type in datatypes:
#        filter.append(data.dtypes == type)
#
#    filtered_columns = data.dtypes[filter].index
#    filtered_data = data[filtered_columns]
#
#    return filtered_data

# Attribute groups
## Because the number of attributes is high, we group them based on the codebook.
## Time
#time_att = ['eventid', 'iyear', 'imonth', 'iday', 'approxdate', 'extended', 'resolution']
## Location
#loc_att = ['country', 'country_txt', 'region', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity','vicinity', 'location']
## Incident
#incid_att = ['summary', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'alternative_txt', 'multiple', 'related']
## Attack
#attack_att = ['success', 'suicide', 'attacktype1', 'attacktype1_txt', 'attacktype2', 'attacktype2_txt', 'attacktype3', 'attacktype3_txt']
## Perpetrators
#perp_att = ['gname', 'gsubname', 'gname2', 'gsubname2', 'gname3', 'gsubname3']
## Perpetrator validity
#perval_att = ['motive', 'guncertain1', 'guncertain2', 'guncertain3', 'individual', 'nperps', 'nperpcap', 'claimed', 'claimmode', 'claimmode_txt', 'claim2', 'claimmode2', 'claimmode2_txt', 'claim3', 'claimmode3', 'claimmode3_txt', 'compclaim']
## Weapon
#weap_att = ['weaptype1', 'weaptype1_txt', 'weapsubtype1', 'weapsubtype1_txt', 'weaptype2', 'weaptype2_txt', 'weapsubtype2', 'weapsubtype2_txt', 'weaptype3', 'weaptype3_txt', 'weapsubtype3', 'weapsubtype3_txt', 'weaptype4', 'weaptype4_txt', 'weapsubtype4', 'weapsubtype4_txt', 'weapdetail']
## Target
#targ_att = ['targtype1', 'targtype1_txt', 'targsubtype1', 'targsubtype1_txt', 'corp1', 'target1', 'natlty1', 'natlty1_txt', 'targtype2', 'targtype2_txt', 'targsubtype2', 'targsubtype2_txt', 'corp2', 'target2', 'natlty2', 'natlty2_txt', 'targtype3', 'targtype3_txt', 'targsubtype3', 'targsubtype3_txt', 'corp3', 'target3', 'natlty3', 'natlty3_txt']
## Casualties and consequences
#cons_att = ['nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte', 'property', 'propextent', 'propextent_txt', 'propvalue', 'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'ransomnote', 'hostkidoutcome', 'hostkidoutcome_txt', 'nreleased']
## Additionl information
#info_att = ['addnotes', 'scite1', 'scite2', 'scite3', 'dbsource', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY',]
#
#att_groups = [time_att, loc_att, incid_att, attack_att, perp_att, perval_att, weap_att, targ_att, cons_att, info_att]

# Grouping numerical and categorcial attributes
# num_att = gtd.dtypes[(gtd.dtypes == 'float64') | (gtd.dtypes == 'int64')].index
# cat_att = gtd.dtypes[(gtd.dtypes == 'object')].index

# The list of the primary perpetrator names
gtd.gname.value_counts()
# The dataset is highly unbalanced. The most frequent primary perpetrator value (around 46%) is "unknown".

# Unaffiliated individual
## The ratio of unaffiliated individuals is 0.2%
gtd.individual.value_counts().iloc[1] / len(gtd)
## There are 314 cases where the perpetrator is an unaffiliated individual but also connected to a vague, broadly defined group.
gtd.gname[(gtd.individual == 1) & (gtd.gname != "Unknown")].value_counts()


# Suspected (i.e. not validated) perpetrators
gtd.guncertain1.value_counts(dropna=False, normalize=True)
gtd.guncertain2.value_counts(dropna=False, normalize=True)
gtd.guncertain3.value_counts(dropna=False, normalize=True)
# 8.4% of primary perpetrators are suspected

# Splitting the dataset based on whether the perpetrators are known
known = gtd[gtd.gname != 'Unknown']
unknown = gtd[gtd.gname == 'Unknown']

# Selecting out groups committing only one terrorist act
oneoff_names = known.gname.value_counts()[known.gname.value_counts() == 1].index
multis = known[~known.gname.isin(oneoff_names)]

# Missing values
# Brute force method
# nonmiss = gtd.dropna(axis=1)
# miss_count = gtd.isnull().sum()[gtd.isnull().any() == True].sort_values(ascending=False)
# miss_rat = gtd.isnull().mean()[gtd.isnull().any() == True].sort_values(ascending=False)
# miss_count.plot(kind='hist')
# nonmiss = miss_rat[miss_rat < 0.9].index


# Correlations
corrs = gtd.corr()

# Correlation matrix
fig = pyplot.figure(figsize=(14,14))
ax = fig.add_subplot(111)
cax = ax.matshow(corrs, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()

# Seeking correlated dimensions
cri_hi = abs(corrs < 1) & abs(corrs >= 0.6)
corr_hi = corrs[cri_hi].stack()
# There are 28 variables highly correlated with each other

# Skews
skews = gtd.skew()
abs(skews).sort_values(ascending=False)

# Visualization
att_nonperp = att_groups[:4] + att_groups[5:]

nonperp = []
for group in att_nonperp:
    for att in group:
        nonperp.append(att)

# Histograms
for att in att_nonperp:
    gtd[att].hist()

# Density plots: I do not use them, because they take too much time to run
# gtd.evenid.plot(kind='density', subplots=True, sharex=False)

# Box plots
gtd[num_att].plot(kind='box', sharex=False, sharey=False, subplots=True, layout=(13,6), figsize=(24,100))

# Scatter matrix
# from pandas.plotting import scatter_matrix
# scatter_matrix(gtd[num_att])

# Splitting Target and Attributes
# Locating the target attribute

clean = gtd.dropna(axis=1)

# First we focus only on the following constraints:
# 1. We only try to predict the primary perpetrator name ('gname')
# 2. We use only those numeric attributes which

att_cl_num = []
for att in clean.dtypes[(clean.dtypes == 'float64') | (clean.dtypes == 'int64')].index:
    #if att in nonperp:
    att_cl_num.append(att)

# nonperp_num = gtd[att_nonperp_num]
# nonpred = clean.columns.drop('gname')
# clean = clean[att_cl_num]

##############################
# Modeling
##############################

X = clean[att_cl_num].as_matrix()
Y = clean.gname.as_matrix()

validation_size = 0.2
seed = 17

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(LogisticRegression(), X_train, Y_train, cv=kfold, scoring='accuracy')
results.append(cv_results)
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

#for name, model in models:
#    kfold = KFold(n_splits=10, random_state=seed)
#    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#    results.append(cv_results)
#    names.append(name)
#    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#    print(msg)
