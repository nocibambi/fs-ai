
# coding: utf-8

# # Importing libraries

# ## Basics

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot


# ## EDA and preprocessing

# In[3]:


from sklearn.preprocessing import Normalizer
from sklearn.ensemble import IsolationForest


# ## Cross validation

# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


# ## Models

# In[5]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.linear_model import LogisticRegression


# ## Metrics

# In[6]:


#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


# # Exploratory data analysis

# Previously we loaded the data and created a sample out of it. From now on we are going to use it for our analysis and modeling.

# In[7]:


#gtd_ori = pd.read_excel("globalterrorismdb_0617dist.xlsx")


# In[8]:


# ecause the dataset is highly imbalanced, I tried to get a stratified sample, but was unable to do so.
#stsmp = train_test_split(gtd_ori, train_size=0.25, startify=['gname'])
#stsmp.to_excel("stsmp.xlsx")
#gtd = pd.read_excel("stsmp.xlsx")

#smp = gtd_ori.sample(frac=0.25, random_state=4721)
#smp.to_excel("sample.xlsx")
gtd = pd.read_excel("sample.xlsx")


# In[9]:


gtd.info(verbose=True, null_counts=True, max_cols=True)


# ### Column data types and the number of unique values in them.

# We list the unique values of attributes and group them by their datatype:

# In[10]:


uniques = gtd.nunique()
types = gtd.dtypes


# In[11]:


atts = pd.concat([uniques, types], axis=1)
#atts.rename(columns=['types', 'uniques'], inplace=True)
atts.columns = ['uniques', 'types']

for coltype in atts.types.unique():
    clist = atts[atts.types == coltype].sort_values(by='uniques', ascending=False)
    print("\n{}:\n\n{}\n".format(coltype, clist.uniques))


# There are many categorical or even binomial attributes which however contain their own missing value codes which we need to recode later.

# ## Attribute domain categories

# Based on the GTD Codebook (adjusted with own analysis), the dataset consists of the following attribute groups:
# 1. Time
# 2. Location
# 3. Incident
# 4. Attack
# 5. Perpetrators
# 6. Perpetrator validity
# 7. Weapon
# 8. Target
# 9. Casualties and consequences
# 10. Additional information

# In[12]:


attgs = {'time' : ['eventid', 'iyear', 'imonth', 'iday', 'approxdate',
                   'extended', 'resolution'],
         'loc' : ['country', 'country_txt', 'region', 'region_txt', 'provstate',
                  'city', 'latitude', 'longitude', 'specificity','vicinity',
                  'location'],
         'incid' : ['summary', 'crit1', 'crit2', 'crit3', 'doubtterr',
                    'alternative', 'alternative_txt', 'multiple', 'related'],
         'attack' : ['success', 'suicide', 'attacktype1', 'attacktype1_txt',
                     'attacktype2', 'attacktype2_txt', 'attacktype3',
                     'attacktype3_txt'],
         'perp' : ['gname', 'gsubname', 'gname2',
                   'gsubname2', 'gname3', 'gsubname3'],
         'perval' : ['motive', 'guncertain1', 'guncertain2', 'guncertain3',
                     'individual', 'nperps', 'nperpcap',
                     'claimed', 'claimmode', 'claimmode_txt',
                     'claim2', 'claimmode2', 'claimmode2_txt',
                     'claim3', 'claimmode3', 'claimmode3_txt', 'compclaim'],
         'weap' : ['weaptype1', 'weaptype1_txt',
                   'weapsubtype1', 'weapsubtype1_txt',
                   'weaptype2', 'weaptype2_txt',
                   'weapsubtype2', 'weapsubtype2_txt',
                   'weaptype3', 'weaptype3_txt',
                   'weapsubtype3', 'weapsubtype3_txt',
                   'weaptype4', 'weaptype4_txt',
                   'weapsubtype4', 'weapsubtype4_txt', 'weapdetail'],
         'targ' : ['targtype1', 'targtype1_txt',
                   'targsubtype1', 'targsubtype1_txt',
                   'corp1', 'target1', 'natlty1', 'natlty1_txt',
                   'targtype2', 'targtype2_txt',
                   'targsubtype2', 'targsubtype2_txt', 'corp2', 'target2',
                   'natlty2', 'natlty2_txt', 'targtype3', 'targtype3_txt',
                   'targsubtype3', 'targsubtype3_txt', 'corp3', 'target3',
                   'natlty3', 'natlty3_txt'],
         'cons' : ['nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus',
                   'nwoundte', 'property', 'propextent', 'propextent_txt',
                   'propvalue', 'propcomment', 'ishostkid', 'nhostkid',
                   'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry',
                   'ransom', 'ransomamt', 'ransomamtus', 'ransompaid',
                   'ransompaidus', 'ransomnote', 'hostkidoutcome',
                   'hostkidoutcome_txt', 'nreleased'],
         'info' : ['addnotes', 'scite1', 'scite2', 'scite3', 'dbsource',
                   'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY',]}


# ### From among these, the 'att_perp' group contains the target variable(s).
# ## Perpetrator names
# In[13]:


gtd[att_perp]


# ### Number of incidents per groups

# More than 45% of the  perpetrators are unknown:

# In[14]:


gtd.gname.value_counts(normalize=True).head(1)


# In[15]:


known = gtd[gtd.gname != 'Unknown']


# ###  Distribution of incidents among groups

# The proportion of recognized groups responsible only for a single incident:

# In[16]:


len(known.gname.value_counts()[known.gname.value_counts() <= 1]) / len(known)


# The distribution of incidents among the known groups:

# In[17]:


inc_grp = known.gname.value_counts()
print(inc_grp)


# Around a fifth  of the groups is responsible for 90% of all _known_ incidents and 40% of them for 95%.

# In[18]:


inc_grp_csum = known.gname.value_counts(normalize=True).cumsum()
inc_grp_csum


# In[19]:


len(inc_grp_csum[inc_grp_csum <= 0.90]) / len(gtd.gname.unique())


# In[20]:


len(inc_grp_csum[inc_grp_csum <= 0.95]) / len(gtd.gname.unique())


# ### Second and third groups

# The database also records second and third groups but only for around 1% of the total incidents:

# In[21]:


names = known.loc[:, ['gname', 'gname2', 'gname3']]


# In[22]:


names.count() / len(known)


# In[23]:


names.apply(lambda x: x.value_counts(dropna=False)).sort_values(by='gname', ascending=False)


# ### Subnames

# The database also records subnames for perpetrators for around 3% of the perpetrators.

# In[24]:


sub_names = gtd.loc[:,['gsubname', 'gsubname2', 'gsubname3']].dropna(how='all').groupby(by=gtd.loc[:,'gname']).count()


# In[25]:


sub_names.sum() / len(gtd)


# In[26]:


sub_names.sort_values(by='gsubname', ascending=False)


# ### Suspected perpetrators

# Around 15% of perpetrator information is in a 'suspected' status.

# In[27]:


guncertcols = gtd.loc[:, ['guncertain1','guncertain2','guncertain3']]


# In[28]:


guncertcols[guncertcols == 1].dropna(how='all').count() / len(known)


# On the other hand, the coding book is not clear about what exactly the '1' of 'uncertainty' value means compared to the '0' and 'NaN' values.

# In[29]:


guncertcols.apply(lambda x: x.value_counts(dropna=False))


# ### Unaffiliated individuals

# The ratio of 'unaffiliated' individuals (i.e. individuals who were recognised but were not affiliated to known groups) is 0.26%.

# In[30]:


gtd.individual.value_counts().iloc[1] / len(gtd)


# Where the perpetrator is an unaffiliated individual it is somehow also connected to a vague, broadly defined group:

# In[31]:


gtd.gname[(gtd.individual == 1) & (gtd.gname != "Unknown")].value_counts()


# ## Summary of perpetrator information analysis
# Because of the relative low coverage, we will not try to predict second/third perpetrators and subnames. Later on, the process might be enhanced to become able to do so.

# ## Special attributes
# The data set also contains a number of special attributes, from among which we drop the `eventid` column, `addnotes` and the information about the record's data source:
# * eventid
# * addnotes
# * scite1
# * scite2
# * scite3
# * dbsource
#
#

# In[32]:


temp = gtd.drop(['eventid', 'addnotes', 'scite1', 'scite2', 'scite3', 'dbsource'], axis=1, errors='ignore')


# ## Missing values

# ### Recoding built-in missing values

# The `miscodes` dictionary defines the attribute-missing value code pairs:

# In[33]:


miscodes = {"0": ['imonth', 'iday'],
            "-9": ['claim2', 'claimed', 'compclaim', 'doubtterr', 'INT_ANY', 'INT_IDEO', 'INT_LOG', 'INT_MISC',
                   'ishostkid', 'ndays', 'nhostkid', 'nhostkidus', 'nhours', 'nperpcap', 'nperps', 'nreleased',
                   'property', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'vicinity'],
            "-99": ['ransompaid', 'nperpcap', 'compclaim', 'nreleased', 'nperps', 'nhostkidus', 'ransomamtus',
                    'ransomamt', 'nhours', 'ndays', 'propvalue', 'ransompaidus', 'nhostkid']}


# In[34]:


temp[miscodes['0']].apply(lambda x: x.value_counts())


# In[35]:


temp[miscodes['-9']].apply(lambda x: x.value_counts())


# In[36]:


temp[miscodes['-99']].apply(lambda x: x.value_counts())


# We transform these missing value codes into numpy `NaN` values

# In[37]:


def mistonan(data, collist, nancode):
    """Replaces columns' missing value code with numpy NaN.

        Parameters:
        `data`: dataframe

        `nanvalue` : the code of the missing value in the columns
        """
    colstonan = []

    for col in collist:
        if col in data.columns:
            colstonan.append(col)
        else:
            print("'{}' is not among the dataframe's columns.".format(col))

    data[colstonan] = data[colstonan].apply(lambda x: x.replace(nancode, np.NaN))


# In[38]:


for key in miscodes.keys():
    mistonan(temp, miscodes[key], float(key))


# We also replace "Unknown" values with numpy `NaN`s whenever it occurs except in the `gname` target attribute (which we do separately, when needed).

# In[39]:


temp.drop(columns='gname').replace(to_replace='Unknown', value=np.NaN, inplace=True)


# ### Missing value ratios

# In[40]:


def missing_ratio(data):
    """
    Lists missing values ratios for each column in a dataset.

    Takes `data`, dataset.

    Returns the `mrat` dataframe, which lists the columns and their corresponding missing value ratios in descending order.
    """
    mrat = data.isna().mean()[data.isna().any()== True].sort_values()

    return mrat


# In[41]:


misrat = missing_ratio(temp)


# From among the total 135 attributes 104 contains missing values. 96 of them has a 95% missing value rate:

# In[42]:


misrat[misrat > 0.05].count()


# In[43]:


misrat


# In[44]:


misrat.hist(bins=20)


# ## Outliers
We examine the dataset for outliers:
# In[45]:


idx_gtd = temp.dtypes[(temp.dtypes == 'float64') |
              (temp.dtypes == 'int64')].index

nums = temp.reindex(idx_gtd, axis=1)

#nums.drop(['eventid'], axis=1, inplace=True)


# In[46]:


nums.plot(kind='box', subplots=True, layout=(26,5), figsize=(25, 100), fontsize=20)


# The dataset is imbalanced by many attributes. The visually recognisable examples:
# * `iyear`
# * `region`
# * `attacktype` series
# * `targtype` and `targsubtype` series
# * `claimmode`
# * `weapontype` and `weaponsubtype`

# Because this is a highly imbalanced dataset we use the `IsolationForest` model to define outliers.

# We drop missing values:

# In[47]:


nums.dropna(axis=1, inplace=True, thresh=nums.shape[0] * 0.95)
nums.dropna(inplace=True)


# In[48]:


clf = IsolationForest(max_samples='auto', random_state=153, contamination=0.05, verbose=True, n_jobs=-1)


# In[49]:


clf.fit(nums)
isof = clf.predict(nums)


# In[50]:


isofdf = pd.Series(isof)
nums['Outlier'] = isof


# In[51]:


nums = nums[nums.Outlier != -1]
nums.drop(columns='Outlier', inplace=True)


# In[52]:


nums.plot(kind='box', subplots=True, layout=(26,5), figsize=(25, 100), fontsize=20)


# # Data preprocessing

# From among the recognised transformation possibilities, the following transformations are 'universal' (that is, could be used for the whole dataset regardless of the particular training/test split method):
# * Remove special attributes
# * Recode built-in missing attributes into numpy NaNs
#
# There are a number of configuration possibilites which we also could use but now we leave untouched:
#
# * Exclude uncertain cases.
# * Include data only for a selected period.
# * Include perpetrators only abouve a particular contribution threshold.
#
# Finally, in the current process we use the following setting:
# * Exclude incidents with 'unknown' perpetrators
# * Include only general names and only of the primary perpetrators (i.e. `gname`)
# * Drop missing values
# * Drop outliers
# * Include only numerical values
# * Exclude 'unaffiliated' individuals.
# * Exclude the `resolution` NaTType attribute

# ## Preprocessing

# We summarized the following configuration possibilities in a function:

# In[53]:


def preproc(data,
            primonly=True,
            period=(1, 5),
            onlyknown=True,
            nocat=True,
            maxna=0.05,
            outrat=0.05,
            topincrat=1,
            hideind=True,
            hideuncert=False,
            dropspec=True,
            dropres=True,
            miscodetonan=True):
    """
    Cleans and preprocesses dataset.

    Parameters:
    ===========

    `primonly`: boolean, True
    Includes only general perpetrator group names and only of the primary perpetrator (`gname`).

    `period`: tuple, (1, 5)
    Defines the included period by setting the start and end dates:
        '1': 1970
        '2': 1998
        '3': April 1 2008
        '4': 2012
        `5`: 2016

    `onlyknown`: boolean, True
    Shows only incidents where the perpetrators is identified (even if with doubt).

    `nocat`: boolean, True
    Excludes all categorical attributes from the dataset with the exception of the perpetrator group name
    attributes (`gname`:`gsubname3`).

    `maxna`: float, 0.05
    The maximum allowed proportion of missing values within an attribute. Drops all rows with missing values and
    keeps only columns with missing value ratio below the given threshold.
    For instance, a value of '0.05' means that only columns with less than 5% of missing values are kept in the dataset.

    `outrat`: float, 0.05
    The contamination ratio determining the percent of values classified as outliers and dropped from the dataset.
    The function uses the Isolation Forest model for identifying outliers.
    Prerequisities:
        * No categorical values (`nocat`)
        * No missing values (`maxna`)
    Example: '0.05' means that 5% of values will be flagged as an outlier and dropped.

    `topincrat`: float, 1
    Filters perpetrators based on their overall weight of contribution (in terms of number of incidents).
    The parameter's value is the ratio of total incidents for which the selected perpetrators are responsible.
    Perpetrators are ranked based on the number of incidents in which they are involved and the function calculates
    their cumulative contribution. It then makes the selection at or right above the given threshold.
    Example: '0.95' means selecting the perpetator groups with the highest incident ratio responsible together
    for 95% of the total incidents.

    `hideind`: boolean, True
    Hides individual perpetrators unaffiliated to groups.

    `hideuncert`: boolean, False
    Hides uncertain cases

    `dropspec`: boolean, True
    Drops special attributes.

    `dropres`: boolean, True
    Drops the `resolution` NaTType attribute.

    `miscodetonan`: boolean, True
    Transforms the original codes for missing values into numpy NaN.
    """
    procd = data.copy(deep=True)

    # `dropspec`: Drop special attributes
    if dropspec == True:
        procd.drop(['eventid', 'addnotes', 'scite1', 'scite2', 'scite3', 'dbsource'], axis=1, inplace=True)

    # `miscodetonan`: Turn built-in missing codes into numpy NaN
    if miscodetonan == True:
        # The `miscodes` dictionary defines the attribute-missing value code pairs:
        miscodes = {"0": ['imonth', 'iday'],
                    "-9": ['claim2', 'claimed', 'compclaim', 'doubtterr', 'INT_ANY', 'INT_IDEO', 'INT_LOG', 'INT_MISC',
                           'ishostkid', 'ndays', 'nhostkid', 'nhostkidus', 'nhours', 'nperpcap', 'nperps', 'nreleased',
                           'property', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'vicinity'],
                    "-99": ['ransompaid', 'nperpcap', 'compclaim', 'nreleased', 'nperps', 'nhostkidus', 'ransomamtus',
                            'ransomamt', 'nhours', 'ndays', 'propvalue', 'ransompaidus', 'nhostkid']}

        def mistonan(data, collist, nancode):
            """Replaces columns' missing value code with numpy NaN.

            Parameters:
            `data`: dataframe

            `nanvalue` : the code of the missing value in the columns
            """
            colstonan = []

            for col in collist:
                if col in data.columns:
                    colstonan.append(col)
                else:
                    print("'{}' is not among the dataframe's columns.".format(col))

            data[colstonan] = data[colstonan].apply(lambda x: x.replace(nancode, np.NaN))

        for key in miscodes.keys():
            mistonan(procd, miscodes[key], float(key))

        # Replaces "Unknown" values whenever it occurs except in the `gname` target attribute.
        # The function controls it by the `onlyknown` parameter.
        procd.drop(columns='gname').replace(to_replace='Unknown', value=np.NaN, inplace=True)

    # Replacing missing values

    ## Replace unknown, '0' days and months with a random value
    ### Months
    #tm = procd.imonth[procd.imonth == 0]
    #tm = tm.apply(lambda x: np.random.randint(1, 13))
    #procd.imonth[procd.imonth == 0] = tm

    ### Days
    #td = procd.iday[procd.iday == 0]
    #td = td.apply(lambda x: np.random. randint(1, 29))
    #procd.iday[procd.iday == 0] = td

    # `period`: Filter the dataset for the choosen time period
    dates = [1970, 1997, 2008, 2012, 2016]

    predmin = dates[period[0]-1]
    predmax = dates[period[1]-1]

    if predmin == 2008:
        procd = procd[((procd.iyear > predmin ) & (procd.iyear < predmax + 1)) |
                      ((procd.iyear == predmin) & (procd.imonth >= 3))]
    elif predmax == 2008:
        procd = procd[((procd.iyear >= predmin ) & (procd.iyear < predmax)) |
                      ((procd.iyear == predmax) & (procd.imonth < 3))]
    else:
        procd = procd[(procd.iyear >= predmin) & (procd.iyear < predmax + 1)]

    # `onlyknown`: Show only known perpetrators
    if onlyknown == True:
        procd = procd[procd.gname != 'Unknown']

    # `hideind`: Hide unaffiliated individuals
    if hideind == True:
        procd = procd[procd.individual != 1]

    # `primonly`: Include only the primary perpetrator groups and only their main names.
    if primonly == True:
        procd.drop(columns=['gsubname','gname2','gsubname2','gname3','gsubname3'], axis=1, inplace=True)

    # `topincrat`: Set the threshold for the top frequent perpetrators to show.
    tempname = procd.gname
    idx_main_groups = tempname.value_counts()[tempname.value_counts(normalize=True).cumsum() <= topincrat].index
    procd = procd[procd.gname.isin(idx_main_groups)]

    # `hideuncert`: Hide uncertain cases
    if hideuncert == True:
        procd = procd[(procd.guncertain1 != 1) |
                      (procd.guncertain2 != 1) |
                      (procd.guncertain3 != 1)]

    # `nocat`: Dropping polynomial attributes (except `gname`)
    if nocat == True:
        idx_nonobj = procd.dtypes[(procd.dtypes.index.isin(['gname',
                                                            'gsubname',
                                                            'gname2',
                                                            'gsubname2',
                                                            'gname3',
                                                            'gsubname3'])) |
                                   (procd.dtypes != 'object')].index

        procd = procd.reindex(idx_nonobj, axis=1)

    # `dropres`: Drop the `resolution` attribute
    if dropres == True:
        procd.drop(columns='resolution', inplace=True)

    #print(procd)

    # `maxna`: Drop missing values and columns with missing values above the threshold.
    procd.dropna(axis=1, inplace=True, thresh=procd.shape[0] * (1 - maxna))
    procd.dropna(inplace=True)

    # `outrat`: Drop outliers
    clf = IsolationForest(max_samples='auto', random_state=2425, contamination=outrat, verbose=True, n_jobs=-1)

    clf.fit(procd.drop(columns='gname'))
    isof = clf.predict(procd.drop(columns='gname'))

    isofdf = pd.Series(isof)
    procd['Outlier'] = isof

    procd = procd[procd.Outlier != -1]
    procd.drop(columns='Outlier', inplace=True)

    # Drop correlated values
    #procd = dropcors(procd, corthr)

    print(procd.info(verbose=True))
    return procd


# Finally we are going to train the model on the following way:
# * Only known perpetrators and only perpetrator groups
# * Only examining the main names of the 'first' perpetrators
# * For the whole time period
# * Dropping attributes:
#     * with more the 5% of missing values
#     * with non-numerical values
#     * `resolution`
# * Dropping 5% of outlier values

# In[54]:


moddat = gtd.copy(deep=True)
moddat = preproc(moddat,
                 primonly=True,
                 period=(1, 5),
                 onlyknown=True,
                 nocat=True,
                 maxna=0.05,
                 outrat=0.05,
                 topincrat=1,
                 hideind=True,
                 hideuncert=False,
                 dropspec=True,
                 dropres=True,
                 miscodetonan=True)


# ### Correlated attributes

# We could not automate the selection of correlated attributes, therefore, we do it them semi-manually:

# In[55]:


def cors(data, threshold=0.5, sort=False):
    """Lists correlation pairs and their correlation values above a correlation threshold.

    `data`: DataFrame

    `threshold`: The correlation value above which it shows the correlation pairs.

    `paired`: True
    Organizes the correlation pairs according to attributes.
    If False, it shows the correlation pairs values in the descending order.
    """

    corrs = data.corr()

    cri_hi = abs(corrs < 1) & abs(corrs >= threshold)
    corr_hi = corrs[cri_hi].stack().reset_index()
    corr_hi.columns = ['first', 'second', 'corr']

    if sort == True:
        output = corr_hi.sort_values(by='corr', ascending=False)
    else:
        output = corrs[cri_hi].stack()

    return output


# In[56]:


corpair = cors(moddat, 0.7, sort=True)
corpair


# In[57]:


moddat.drop(columns=['extended', 'targsubtype1', 'INT_LOG', 'INT_IDEO'], inplace=True, errors='ignore')


# # Modeling

# In[58]:


X = moddat.drop(['gname'], axis=1).dropna(axis=1)
X.dropna(axis=1, inplace=True)
print(X.shape)


# Normalizing the dataset:

# In[59]:


scaler = Normalizer().fit(X)
X = scaler.transform(X)


# In[60]:


y = moddat.gname
y.dropna(inplace=True)
y.fillna("NaN", inplace=True)
y.shape


# In[61]:


validation_size = 0.2
seed = 17


# In[62]:


X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)
print(X_train.shape)
print(X_validation.shape)
print(y_train.shape)
print(y_validation.shape)


# In[63]:


results = []
names = []


# In[64]:


kfold = KFold(n_splits=10, random_state=seed)


# ## Cross validating models

# Finally I decided to focus only on Decision Tree and K-NN classifiers for learning and performance reasons:

# In[65]:


models = {"Decisiong Tree Classifier": DecisionTreeClassifier(),
          "K-Neighbors Classifier": KNeighborsClassifier()}

# "Linear Discriminant Analysis": LinearDiscriminantAnalysis()
# "Logistic Regression": LogisticRegression()


# In[66]:


def predict_groups(models, X_train, y_train):
    for model in models:
        #print("\n{}:\n\n{}\n".format(model, models[model]))

        model_score = cross_val_score(models[model], X_train, y_train, cv=kfold, scoring='accuracy')
        print("\n{}:\n\tAccuracy: {} ({})".format(model, model_score.mean(), model_score.std()))

        model_score = cross_val_score(models[model], X_train, y_train, cv=kfold, scoring='f1_micro')
        print("\tF1 micro: {} ({})".format(model_score.mean(), model_score.std()))

        #model_score = cross_val_score(models[model], X_train, y_train, cv=kfold, scoring='f1_weighted')
        #print("\tF1 weighted: {} ({})".format(model_score.mean(), model_score.std()))

        #model_score = cross_val_score(models[model], X_train, y_train, cv=kfold, scoring='roc_auc')
        #print("\tF1 ROC: {} ({})".format(model_score.mean(), model_score.std()))

        #crosval = cross_validate(model, X, y, scoring=['accuracy', 'precision_micro', 'recall_micro', 'f1_micro'])



# In[67]:


predict_groups(models, X_train, y_train)


# The two models produce prediction accuracies of around 68% and 60% respectively with minimal standard deviations. The F1 scores are almost the same (which is normal for multiclass classification problems).
#
# While this is not necessarily bad, it is tested only on a selected dataset and therefore should be developed further.

# # Possible enhancements for the process
#
# ## Target variables
# * Taking into account information about second and third perpetrators.
# * Taking into account subname information.
#
# ## Data preprocessing
# * Tuning the selected data for training based on the possible parameters already identified
# * Identifying the strongest predictor attributes in the dataset and trying to predict those with date we excluded now (e.g. records with 'Unknown' perpetrators).
# * Handling imbalanced features, most importantly dates
# * Recoding and using categorical data
# * Including also textual data.
# * Testing the models also on the whole dataset
#
# ## Models tuning
# * Tuning model hyperparameters (e.g. manually or with grid or random search)
#
