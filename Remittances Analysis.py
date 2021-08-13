#!/usr/bin/env python
# coding: utf-8

# Import required libraries

# In[21]:


import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Create functions to simplify searching the world bank database and creating charts

# In[8]:


#this is the list of functions for this project
def search_database(string):
    #this cycles through all the databases searching for a string
    for i in range(1,70):
        try: 
            wb.db = i
            found = wb.series.info(q=string)
            print(f"In database {i} I found these series {found}")#" and the length of list is {leng}")
        except:
            print(f"nothing found in database {i}")
    print("end of loop")

def pull_data(database,indicator,countries):
    #simplifies pulling a specific indicator
    df_spec = "df" + str(indicator)
    wb.db = database
    df_spec = wb.data.DataFrame(indicator, countries, time=range(1991, 2021))
    df_spec = df_spec.T
    df_spec.fillna(method="ffill",inplace=True)
    print(df_spec.tail())
    return(df_spec)

def pretty_hist(mydataframe):
    #creates a pretty histogram
    hist = mydataframe.iloc[-1].hist(bins=20)
    median = mydataframe.iloc[-1].median()
    plt.axvline(x=median,color="r")
    print(hist)

def merge_data(string_x,string_y,series_x,series_y):
    #concats and plots a pretty scatterplot
    df = pd.concat([series_x, series_y], axis=1,join="inner")
    df.columns = [string_x,string_y]
    print(df.head())
    return(df)

def pretty_scatter(string_x,string_y,df):
    fig, ax = plt.subplots(figsize=(8,5))
    plt.scatter(df[string_x],df[string_y])
    ax.set_xlabel(string_x)
    ax.set_ylabel(string_y)
    plt.show()
    return(df)

print("functions loaded")


# Create dictionaries to use later to rename acronyms

# In[9]:


#this prints all the relevant economies and has latitude and longitude data
df_allcountries = wb.economy.DataFrame(skipAggs=True)

#this stores all the country tickers into allcountries variable
allcountries_dict = df_allcountries.name.to_dict()
allcountries = allcountries_dict.keys()

#this creates a dictionary for regions
allcountries_regions = dict(zip(df_allcountries.index, df_allcountries.region))

#this creates a dictionary for aggregate regions
region_dict = {"LCN":"Latin America & Caribbean","SAS":"South Asia","SSF":"Sub-Saharan Africa","ECS":"Europe & Central Asia","MEA":"iddle East & North Africa","EAS":"East Asia & Pacific","NAC":"North America"}


# Start pulling remittance data and inspecting

# In[10]:


#pull remittance data and create a histogram
df_remittances = pull_data(2,"BX.TRF.PWKR.DT.GD.ZS",allcountries_dict.keys())  
print("These are the 20 largest remittance recipients as % of GDP")
print(df_remittances.iloc[-1].nlargest(20).rename(allcountries_dict))


# In[11]:


#pull GDP per capita data and create a histogram
df_gdp_pc = pull_data(2,"NY.GDP.PCAP.CD",allcountries_dict.keys()) 
pretty_hist(df_gdp_pc)

df_remit_gpc = merge_data("GDP per capita","Remittances as % of GDP",
               df_gdp_pc.iloc[-1],df_remittances.iloc[-1])
df_remit_gpc_s = pretty_scatter("GDP per capita","Remittances as % of GDP",
               df_remit_gpc )
df_remit_gpc["bins"] = pd.cut(df_remit_gpc["GDP per capita"],bins=range(0,110000,10000))#,labels=range(100000,110000))
df_remit_gpc_bins = df_remit_gpc.groupby("bins").mean()
df_remit_gpc_bins["Remittances as % of GDP"].plot(kind="bar")
print("Lower income countries tend to be larger recipients of remittances")


# In[12]:


#filter out richer countries
df_gdp_pcT  = df_gdp_pc.T
df_gdp_pc_filtered = df_gdp_pcT[df_gdp_pcT["YR2020"] < 10000].T
pretty_hist(df_gdp_pc_filtered)


# In[13]:


#filter out outliers in remittances
df_remittancesT  = df_remittances.T
df_remittances_filtered = df_remittancesT[df_remittancesT["YR2020"] < 30].T
pretty_hist(df_remittances_filtered)


# In[22]:


#plot remittances ad a function of GDP per capita
df1 = merge_data("GDP per capita","Remittances as % of GDP",
               df_gdp_pc_filtered.iloc[-1],df_remittances_filtered.iloc[-1])
sns.regplot(x="GDP per capita", y="Remittances as % of GDP", data=df1)
print("""""")
print("There is not a clean relationship between GDP per capita and remittance flows")


# Another likely driver of remittances is the size of a country's diaspora. We can calculate a migration ratio by taking net migration data and adjusting by population.

# In[23]:


#pulls data for net emigration and total population
df_migration = pull_data(2,"SM.POP.NETM",allcountries)
df_population = pull_data(2,"SP.POP.TOTL",allcountries)
#this is the total number of migrants over last x years, a positive numbers is ppl leaving the country
totalmigration = df_migration.sum(axis=0) / 5
averagepopulation = df_population.mean(axis=0)
migration_ratio = totalmigration/averagepopulation * -100
# df1 merge_data("Net migration","Remittances as % of GDP",
#                migration_ratio,df_remittances.iloc[-1])


# In[25]:


#filtering the above data
migration_ratio_filtered = migration_ratio[migration_ratio.between(0, 40)] # & migration_ratio > -10]
df_migration = merge_data("Net migration % of total population","Remittances as % of GDP",
               migration_ratio_filtered,df_remittances.iloc[-1])
df_migration.rename(index=allcountries_dict,inplace=True)

sns.regplot(x="Net migration % of total population", y="Remittances as % of GDP", data=df_migration)
print("""""")
print("Again, there is some correlation, but not exactly a great fit")


# Liuckily, the world bank conducts a survey on remittance habits around the world. We can access that database and see if there arny interesting correlations.

# In[34]:


wb.economy.info(db=28)
remittance_survey = wb.series.Series(q="remittances")
remittance_survey_list = remittance_survey.index
df_survey = pull_data(28,remittance_survey_list,allcountries)


# In[40]:


df_survey.head()
surveys = pd.DataFrame()
for item in remittance_survey_list:
    subset = df_survey.iloc[:, df_survey.columns.get_level_values(1)==item].iloc[-1]
    subset_list = subset.to_list()
    surveys[item] = subset_list

# Compute the correlation matrix
corr = surveys.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 10))
#cmap = sns.diverging_palette(400, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap="BrBG", center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": 1})


# The suervey question that we're interest in is
# "fin26.28.t.a      Sent or received domestic remittances in the past year (% age 15+)"
# and so we can correlate other questions to that one.
# 
# 
# 
# 
# 

# In[41]:


corr = corr.sort_values(by="fin26.28.t.a")
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, mask=mask, cmap="BrBG", center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": 1})


# In[42]:


#convert question codes to actual question for readability
#dict_survey = dict(zip(remittance_survey.index,remittance_survey.values))
#corr = corr.rename(index=dict_survey,columns=dict_survey)
#pd.options.display.max_rows = 100
print(corr.iloc[:20,0])


# The series answer most uncorrelated with having sent remittances over the past year is "Received domestic remittances: in person and in cash only (% recipients, age 15+)". Let's explore that. 

# In[43]:


#compare the two series
df_remit_survey2 = pull_data(28,"fin27c1.t.d.s",allcountries)
df_remit_survey= pull_data(28,"fin26.t.a",allcountries) 
x = "received in last year"
y = "received in person only"
df_survey_comp = merge_data(x,y,df_remit_survey.iloc[-1],df_remit_survey2.iloc[-1])
sns.regplot(x=x, y=y, data=df_survey_comp)
print("")
print("If a person reports having only received remittances in person or in cash only, they are less likely to have received remittances in the past year.")


# In[52]:



# dict_survey = dict(zip(remittance_survey.index,remittance_survey.values))
# corr = corr.rename(index=dict_survey,columns=dict_survey)
pd.options.display.max_rows = 100
print(corr.iloc[-50:,0])

#pd.options.display.max_rows = 100
#Received domestic remittances: through a mobile phone (% recipients, age 15+) fin27b.t.a
#Sent or received domestic remittances: using an account (% age 15+)  fin27a.t.d


# In[61]:


#compare the two series
df_remit_survey= pull_data(28,"fin26.t.a",allcountries) 
x = "received in last year"
df_remit_survey2 = pull_data(28,"fin27b.t.a",allcountries)
y = "received using a mobile phone"
df_survey_comp = merge_data(x,y,df_remit_survey.iloc[-1],df_remit_survey2.iloc[-1])
sns.regplot(x=x, y=y, data=df_survey_comp)
print("")
print("Mobile phone technology tends to be related with more remittance.")


# In[62]:


#compare the two series
df_remit_survey= pull_data(28,"fin27c1.t.d",allcountries) 
x = "received in last year"
df_remit_survey2 = pull_data(28,"fin27b.t.a",allcountries)
y = "received using a mobile phone"
df_survey_comp = merge_data(x,y,df_remit_survey.iloc[-1],df_remit_survey2.iloc[-1])
sns.regplot(x=x, y=y, data=df_survey_comp)
print("")
print("Mobile phone technology tends to be related with more remittance.")


# In[ ]:





# In[ ]:





# Look at outliers and see whats special about them...

# In[ ]:





# In[ ]:


# #create a mutlicolor scatterplot based on regions

# x="Net migration % of total population"
# y="Remittances as % of GDP"

# df_217['regions'] = df_217.index.to_series().map(allcountries_regions)
# regionslist=['LCN', 'SAS', 'SSF', 'ECS', 'MEA', 'EAS', 'NAC']
# colorlist=['r','g','b','c','m','y','k']
# for count, region in enumerate(regionslist, start=0):
#     df_217_f = df_217[df_217["regions"] == region]
#     plt.scatter(df_217_f[x],df_217_f[y],color=colorlist[count])
   


# ax.set_xlabel(x)
# ax.set_ylabel(y)
# # plt.xlim=[0,10]

