import pandas as pd

def dropcors(data, thresh=0.8):
    """Drops correlated attributes.
    1. Takes a dataframe and a threshold.
    2. Calculates correlations inbetween columns.
    3. First drops those columns which are correlated with more than one column.
    4. Then sorts the remaining column first by their name's length and then alphabetically
    (this supposed to non-perfectly mirror the naming logic of the dataset).
    Starting from the top it drops its pair from the bottom.

    Parameters
    ----------
    data : dataframe

    thresh : numeric, 0.8
    The threshold above which the function identifies and drops correlated columns.
    """
    def halfcors(dat):
        """Finds reverse duplicates of correlation pairs and drops them.
        """
        halved = []

        for i in range(len(dat)):
            revpair = (dat.iloc[i,1], dat.iloc[i,0])

            if revpair in halved:
                pass

            else:
                halved.append((dat.iloc[i,0], dat.iloc[i,1]))

        return halved


    def listpairs(pairslist):
        """Lists all the elements in the correlations pairs"""
        countatt = []

        for pair in pairslist:
            countatt.append(pair[0])
            countatt.append(pair[1])

        return countatt


    def dropdup(pars, dups):
        """Dropping selected pairs from the list of correlated pairs"""
        for dup in dups:
            ind = pars[pars == dup].index
            pars.drop(ind)

        return pars

    #print("\n\nCurrent columns in data at the beginning:\n\n{}".format(data.columns))

    corr_preproc = data.corr()
    cri_hi_prep = abs(corr_preproc < 1) & abs(corr_preproc >= thresh)

    atts_corr = corr_preproc[cri_hi_prep].stack().reset_index()
    atts_corr.columns=['first', 'second', 'corr']
    print(len(atts_corr))
    print("\nCorrelation pairs:\n\n{}".format(atts_corr))

    halfpars = halfcors(atts_corr)
    #print(len(halfpars))
    #print("\n\nhafpars:\n\n{}".format(halfpars))

    count_att = listpairs(halfpars)
    #print(len(count_att))
    #print("\n\ncount_att:\n\n{}".format(count_att))

    coratrank = pd.Series(count_att).value_counts()
    #print(len(coratrank))
    #print("\n\ncoratrank:\n\n{}".format(coratrank))

    # Recording attributes which correlate with more than one another attribute.
    drpat = []

    #for at in coratrank[coratrank > 1].index:
    #    drpat.append(at)

    #print(len(drpat))
    #print("\n\ndrpat (first):\n\n{}".format(drpat))

    countattS = pd.Series(count_att)
    sings = sorted((dropdup(countattS, drpat).str.lower()), key=lambda x: (len(x), x))
    #print(len(sings))
    #print("\n\nsings (first):\n\n{}".format(sings))

    for sing in sings:
        for i in halfpars:

            if i[0] == sing:
                drpat.append(sing)
                if i[1] in sings:
                    sings.remove(i[1])

            if i[1] == sing:
                drpat.append(sing)
                if i[0] in sings:
                    sings.remove(i[0])

    print(len(drpat))
    print("\nRemove the following {} columns:\n\n{}".format(len(drpat), drpat))

    wocorrs = data.drop(columns=drpat)

    print("\nRemaining columns:\n{}\n{}".format(len(wocorrs.columns), wocorrs.columns))

    return wocorrs
