# Official Problem description
"Use attack type, weapons used, description of the attack, etc. to build a model that can predict what group may have been responsible for an incident."

## What is the problem?
### Informal description
We have data about terrorist attacks but about many of them we do now know who commited them. Our model should help us to tell who might been behind these.

### Formal description
* Experience: Data about previous terrorist attacks.
* Task: Classify a terrorist attack into a terrorist group.
* Performance: The number of terrorist acts classified accurately.

### Assumptions
* There are clearly defined groups behind the indicent.
* Terrorist groups have a robust and consistent method which does not change significantly.
* Terrorists approach "terror problems" similarly across groups, regions and time periods and therefore the patterns of some terrorist acts can help us to explain others or predict future ones.
* Particular definition of terrorism (see appendixes)

### Similar problems
Domain similarity: Other violent but non-terrorist acts

## Why does the problem needs to be solved?
### Motivation
* Violent acts, like terrorism are wrong and should be stopped.
* Better understanding terrorism might help us to also understand the motivation of the perpetrators and the broader circumstances causing them.

### Possible benefits
* Having a better picture of the working of terrorist groups
* Identify common patterns among terrorist acts and their perpetrators

### Solution use
* We can estimate better unknown past and future perpetrators.
* Predict and prevent future events from happening.
* Identifying organizing principles behind terrorist acts and groups.

## How would I solve the problem without machine learning?
* Ethnographic approach: Interviewing perpetartors and their peers to gain knowledge about the story of their terrorist projects. What were their motivations, what were their aims, what circumstances did they have to follow, what practicalities did they need to attend?
* Macrosociological approach: Examining the actual socio-economic patterns within the region preceding the terrorist acts and the material-technological circumstances with which the perpetrators needed to work with.

# The Data
The project examines the following dataset:
> National Consortium for the Study of Terrorism and Responses to Terrorism (START). (2016). Global Terrorism Database [Data file]. Retrieved from https://www.start.umd.edu/gtd

Data is missing for 1993.
```python
gtd.iyear.value_counts().sort_index()
```
Nonetheless, these are also available in a separate file.

## GTD history timeline
* 2001 University of Mariland gains data from Pinkerton about terrorism events from 1970 to 1997 (GTD1)
* 2005 Digitization: corrections and adding information
* April 2006 Funding to extend the data beyond 1997 through 2007 from archival sources and with a different idea of terrorism (GTD2)
* 2008 data collection is finished, applying the new inclusion criteria also on the previous data
* Spring of 2008- Spring of 2012: ISVG collects data betwen April 2008 - October 2011. This is integrated and the existing data is improved.
* 2012 - Data starting with November 2011 is conducted by START,
    - "As a result of these improvements, **a direct comparison between 2011 and 2012 data likely overstates the increase in total attacks and fatalities** worldwide during this time period."

**Users should note that differences in levels of attacks before and after January 1, 1998, before and after April 1, 2008, and before and after January 1, 2012 may be at least partially explained by differences in data collection;**

Cases when the **"coder noted some uncertainty whether an incident meets all of the criteria for inclusion ("Doubt Terrorism Proper,")"**

The GTD includes failed attacs, but does not include foiled or failed plots and violent threats where no action were taken.
No state terrorims is included.

sources range from well-known international news agencies to English translations of local papers written in numerous foreign languages.

Prior to 2012, identifying a year’s worth of terrorist incidents for inclusion in the GTD typically involved the use of approximately 300 unique news sources. By comparison, the 2012 update is based on a pool of over 1500 unique news sources.

# Notes from the codebook
## Criterial for terrorism
Three main criteria of terrorism
- intentional
- entails violence (property/people)
- sub-national actor

In addition, at least two of the following:
- gave some social goal (i.e. not only for profit)
- intention to convey message to the broader public
- outside humanitarian law

## Attributes

### Date
- The `imonth` and `iday` attributes have a 0 value for unknown dates
    - either drop them or take into account as information(?)
    - examine their relationship with the `approxdate` attribute
- Examine what can be done with the extended incident resolution date (`resolution`)
    - instead of simple date, calculate the duration of the incident

### Incident information
- multi categorical
    - `incident`
    - `alternative`: should not this be rather 5 different categorical?
- text
    - `alternative_txt`
- `doubterr`: -9 means not included at the time of data collection
    - drop?
- `multiple` and `related`: try to connect the related incidents together

### Location
-`country` and `country_txt`: there is an 'unknown' country
    - break it up into binomials?
- text
    - `country_txt`
    - `region_txt`
    - `provstate`
    - `city`
    - `location`
- multi categoricals
    - `country`
    - `region`
    - `specificity`
- `latitude` and `longitude`: think/read about how to use them

### Attack type
multi categoricals
- `attacktype1`
- `attacktype2`
- `attacktype3`

text
- `attacktype1_txt`
- `attacktype2_txt`
- `attacktype3_txt`

### Weapon information
multi categoricals
- `weapontype1`
- `weapontype2`
- `weapontype3`
- `weapontype4`
- `weaponsubtype1`
- `weaponsubtype2`
- `weaponsubtype3`
- `weaponsubtype4`

- There are 'other' and 'unknown' values.
- there is a hierarchical relationship between weapon types and subtypes

text
- `weapontype1_txt`
- `weapontype2_txt`
- `weapontype3_txt`
- `weapontype4_txt`
- `weaponsubtype_txt1`
- `weaponsubtype_txt2`
- `weaponsubtype_txt3`
- `weaponsubtype_txt4`
- `weapondetail`

- there is more than one attack type only when it considers a sequence of events
- the attack coded hierarchically: what can I do with it?
- there is an unknown category
- instead of success and occurance we might rather focus on casualties and caused property damage

### Target/victim
multi-categorical
- `targtype1`
- `targtype2`
- `targtype3`
- `targsubtype1`
- `targsubtype2`
- `targsubtype3`
- `natlty1`
- `natlty2`
- `natlty3`

- There are 'unknown' values for the targtype but for the subtype there is none

text
- `targtype1_txt`
- `targtype2_txt`
- `targtype3_txt`
- `targsubtype1_txt`
- `targsubtype2_txt`
- `targsubtype3_txt`
- `corp1`
- `corp2`
- `corp3`
- `target1`
- `target2`
- `target3`
- `natlty1_txt`
- `natlty2_txt`
- `natlty3_txt`

### Perpetrator informaton
text
- `gname`
- `gname2`
- `gname3`
- `gsubname`
- `gsubname2`
- `gsubname3`
- `claimmode_txt`
- `motive`

If I include the 2. and 3. perpetrator information, I might connect the certainty to them.

multi-categoricals
- `claimmode`
- `claimmode2`
- `claimmode3`

- `nperps`, `nperpcap`, `complaim`:
    - contain '-99' or 'unknown'
    - they show the minimum valid number

### Casualties and consequences
`property`, `ishostkid`, `ishostkidus`, `nhostkid`, `nhostkidus`, `nhours`, `ndays`, `ransom`, `ransomamt`, `ransomus`, `ransomamtus`, `ransompaid`, `ransompaidus`, `nreleased`
-9 or -99 stands for unknown

related
- `nhours` and `ndays`
- `divert`and `kidhijcountry`

text
- `propextent_txt`
- `propcomment`
- `divert`
- `kidhijcountry`
- `ransomnote`
- `hostkidoutcome_txt`

multi-categoricals
- `propextent`
- `hostkidoutcome`

`propvalue`: if empty, does not mean there is no damage, but that it is not possible to estimate

### Additional informaton and sources
text
- `addnotes`
- `scite1`
- `scite2`
- `scite3`
- `dbsource`

multi-categorical or unknown
- `INT_LOG`
- `INT_IDEO`
- `INT_MISC`
- `INT_ANY`
