#!/usr/bin/env python
# coding: utf-8

# # PS 88 Project Part 3 - Due 12/14 at 11:59pm

# ## Step 1: Theory
# 
# What is the theoretical question or causal relationship you aim to explore with your analysis? What relationships did you expect to see in the data? (5-10 sentences)

# *The Conscription of Wealth: Mass Warfare and the Demand for Progressive Taxation* by Kenneth Scheve and David Stasavage (2010) showed how progressive taxation rose after WW1 and concluded that mass mobilization for warfare caused increasing demands in progressive taxation. Through this analysis, we want to analyze how progressive taxation is influenced during periods of relative peace -- when there is no mass mobilizing warfare. What factor -- other than mobilization for war -- causes fluctuation in high income tax rates over time? In order to answer this question we needed data from more countries. However, the data available to us is regarding inheritance tax rates not progressive tax rates.
# 
# Hence, throughout this analysis, the first question we ask ourselves is: Given our data, can we use inheritance tax rates as a proxy for all taxation?
# 
# If the answer to the above question is yes, then: Is ideology the reason behind the decline of tax rates post 1950?
# 
# We expect the answer to the latter question to be affirmative. Most importantly, in accordance with common conception of political ideologies, we expect our analysis to show us a positive relationship between left executive and high tax rates. 

# ## Step 2: Merging
# 
# Load up the data files you plan to use, and merge them together. Explain what you are doing at each step of the process. Do some checks with `.shape` to see that the merge works as expected. Depending on what you are working with, this will probably take around 10 lines of code, and 1-2 sentences explaining each step.

# In[1]:


# Run this cell to import the packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16,8)
plt.rcParams['figure.dpi'] = 150
sns.set()
pd.set_option('display.max_columns', None)


# **Part 1: Importing Data**
# 

# We will first import the data from *The Conscription of Wealth: Mass Warfare and the Demand for Progressive Taxation* (Scheve and Stasavage 2010) under the name ss_2010, then data from *Replication Materials for ‘Democracy, War, and Wealth Lessons from Two Centuries of Inheritance Taxation*  (Scheve, Kenneth, David Stasavage 2012) under the name ss_2012.

# In[2]:


data_string = "data/Scheve_Stasavage_IO_2010_CoWreplicationdata.csv"
ss_2010 = pd.read_csv(data_string)
ss_2010.head()


# In[3]:


data_string = "data/Scheve_Stasavage_APSR_2012_inheritannual.csv"
ss_2012 = pd.read_csv(data_string)
ss_2012.head()


# Below is information about the new data.

# In[4]:


data_string = "data/Scheve_Stasavage_APSR_2012_Readme.txt"
ss_2012_description = open(data_string).readlines()
for i in ss_2012_description: 
    print(i)


# **Part 2: Merging the Dataframes**

# In order to merge the two dataframes together, we need to gather more information about them.

# In[5]:


[ss_2010["country"].unique(), ss_2012["name"].unique()]


# In[6]:


ss_2010.shape


# In[7]:


ss_2012.shape


# In[8]:


ss_2012.info(), ss_2010.info()


# The primary key for both dataframes is the country code (ccode) and year. Common features about countries between the two dataframes are country names and war mobilization (himobpopyear2p and himobpopyearp). Hence, we will merge our dataframes on country code, year, country name, himobpopyear2p, himobpopyearp. In order to merge on these features, however, we must first standardize how country names are formatted and the name of the columns. 

# The country names in the new data are written in all lower case letters. Below, we transform country names in the data from the 2010 paper to all lower case. 

# In[9]:


ss_2010["country"] = ss_2010["country"].str.lower()


# The column names for country names in both dataframes are different. We need to match the column names in order to merge on country names. Below, we rename the column name in the ss_2012 dataframe from name to country.

# In[10]:


ss_2012 = ss_2012.rename(columns = {"name": "country"})


# Below, we merge the two dataframes. We do an outer join in order to encompass all data.

# In[11]:


merged_table = pd.merge(ss_2010, ss_2012, on = ["ccode", "year", "country", "himobpopyear2p", "himobpopyearp",
                                                    ], 
                        how = "outer")
merged_table


# ## Step 3: Analysis
# 
# Perform you new analysis. Interpret any graphs or regression output. How do the results change compared to the original paper/lab? This will probably take about 10-15 lines of code, and again provide 1-2 sentences explaining why you are doing what you do and explaining the results.

# **Part 1**
# 
# 
# **Progressive Tax Rate vs. Inheritance Tax Rate: Can Inheritance Tax Rate be a Proxy to Taxation Policy in General?**

# Our original paper analyzed the effect of mass mobilization for war on progressive tax rate. However, the new data we imported analyzes inheritance tax. So, firstly we want to analyze how inheretance tax rate and progressive tax rate changed over time on the same plot.
# 
# However, data on progressive tax rate only exists for countries from the first paper. Hence, in order to do a true comparison -- for this first part -- we will limit our dataframe to the countries that are common in both papers. 

# In[12]:


common_countries = merged_table[(merged_table["country"]).isin(["usa", "netherlands",
                                             "canada", "uk", "france", "sweden", "japan"])]
common_countries.head()


# We will add a column to common_countries that is the difference between Progressive Tax Rate (topratep) and Inheritance Tax Rate (topitaxrate2).

# In[13]:


common_countries["difference_in_tax_rates"] = common_countries["topratep"] - common_countries["topitaxrate2"]


# In[14]:


common_countries["topratep"] - common_countries["topitaxrate2"]


# In[15]:


sns.lineplot(x = "year", y= "difference_in_tax_rates", data = common_countries, ci = False)
plt.title("Figure 1: Difference between Progressive Tax Rate and Inheritance Tax Rate")
plt.ylabel("Tax Rate Difference");


# In[16]:


sns.lineplot(x = "year", y= "topratep", data = common_countries, label = "Progressive Taxation", palette = "Set1", ci = False)
sns.lineplot(x = "year", y= "topitaxrate2", data = common_countries, label = "Inheritance Taxation", palette = "Set1",ci = False)
plt.title("Figure 2: Evolution of Progressive vs. Inheritance Taxation over Time")
plt.ylabel("Tax Rate");


# The evolution of progressive and inheritance tax rates follows a similar trend. However, figure 1 shows that the difference between tax rates rise post WW1. This indicates that the inheritance tax rate isn't as affected by mass mobilization for warfare during WW1. To test if the causative relationship between mass mobilization for warfare and progressive tax rate exists between mass mobilization for warfare and inheritance tax, we will perform regression.

# Below, we replicate a version of the regression we did in part 2 of the project in order to continue the above comparison between progressive tax rate and inheritance tax rate.
# 
# The dependent variable is top tax rate and independent variables are mobilization for war, male universal suffrage, proportion of left seats, GDP per capita, revenue to GDP, democracy, direct election, left executive and military expenditure.

# In[17]:


smf.ols('topratep ~ wwihighmobaft + munsuff + leftseatshp + gdppcp + ratiop + democracy + directelec + leftexec2 + Rmilexbadjdol',
        data=common_countries).fit().summary()


# Below we perform the same regression as above, but we change our dependent variable to inheritance tax rate.

# In[18]:


smf.ols('topitaxrate2 ~ wwihighmobaft + munsuff + leftseatshp + gdppcp + ratiop + democracy + directelec + leftexec2 + Rmilexbadjdol',
        data=common_countries).fit().summary()


# Apart from mass mobilization for warfare, most of the other features have similar coefficients (within each others' standard deviation) for both regressions. Hence, we can assume that for the second half of the 20th century inheritance taxation is a sufficient proxy in understanding the evolution of tax rates.

# **Part 2**
# 
# **Political Ideology and Taxation**

# We want to analyze if a decrease in leftist ideology in the second part of the 20th century (fueled by Cold War) can be associated with the stagnation and decrease in high income and inheritance tax rates. 
# 
# Reminder: we are using inheritance tax rates as a proxy to all high income tax rates.

# Below, we create a new dataframe called post_1950 that includes data on all countries in the merged_table for years after 1950 (1950 included).

# In[19]:


post_1950 = merged_table[merged_table["year"]>1949]
post_1950.head()


# The below code is to analyze the different values for leftexec2.  

# In[20]:


pd.crosstab(post_1950["country"], post_1950["leftexec2"])


# Switzerland is the only country with nonbinary values attached to leftexec2. In order to avoid any errors that might have happened during data entry and in order to make the figure below easier to read, we will remove Switzerland from our analysis.

# In[21]:


sns.boxplot(x = "leftexec2", y = "topitaxrate2", data = post_1950[post_1950["country"]!="switzerland"], 
            palette = "pastel")
plt.xlabel("Left Executive")
plt.ylabel("Top Inheritance Tax Rate")
plt.title("Figure 3: Distribution of Top Inheritance Tax Rate according to the Ideology of the Government");         


# The box plots show that the median tax rate is 10 points lower for left-wing governments compared to right wing governments. They illustrate that in general right executives are associated with high tax rates. This finding contradicts common belief. We will further analyze it with the regression below.
# 
# In the regression below, top inheritance tax rate is our dependent variable and left executive is our main independent variable. We use war mobilization, GDP per Capita and military expenditure as control variables.

# In[22]:


smf.ols('topitaxrate2 ~ leftexec2 + himobpopyear2p + himobpopyearp + gdppc + Rmilexbadjdol',
        data=post_1950).fit().summary()


# The above regression also illustrates a clear negative association between left executive and top tax rates for the period after 1950.

# ## Step 4: Conclusion
# 
# What did you learn form this exercise? How would you extend or modify your analysis if you had more time/data available?  (5-10 sentences)

# When we first embarked on this analysis, we believed that the decline in high income and inheritance tax rates in the second half of 20th century would be strongly associated with the decline of leftist ideology. However, our analysis showed that, contrary to common belief, left executives were associated with lower high income tax rates. 
# 
# However, we believe that our data is insufficient to actually conclude that there is a negative causation between left leaning governments and high income taxation. There is simply too many confounding variables and the dataset is limited to the same kind of countries.
# 
# As of know, our analysis doesn't contradict any of the findings in the original paper. On the contrary, it reinforces them. 
# 
# However, we want to truly understand how taxation is influenced in times of no mass mobilization. Hence, if we had more time and resources we would extend our analysis to all countries. We would gather data on progressive tax rates, type of regime, ideology of government, gdp per capita, military expenditure, tax revenue per capita on all countries around the world starting from 1950 to 2020. We understand that the cold war and the existence of communist regimes might make it impossible to find this data. However, we should expand our dataset.
# 
# Finally, this exercise showed us the importance of performing qualitative research alongside quantitative research. During this post WW2 period, it would be interesting to more deeply analyze the particular moments that led to policy decisions in certain countries to cut down high income taxation. Was it due to lobbying groups? Domestic pressure? Change in political party? All of these questions can be answered substantively thorough qualitative research on certain case studies.

# In[ ]:




