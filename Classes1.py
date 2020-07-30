#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno

class Visualization():
    def __init__(self,data):
        self.df = data
    
    def scatter(self,x,y,z = None):
        scatter = sns.scatterplot(x = x, y = y, hue=z,data=self.df)
        return scatter
    
    def hist(self,x):
        plt.figure(figsize=(8,6))
        hist = sns.distplot(self.df[[x]], rug=True, rug_kws={"color":"g"}, kde_kws={"color":"k","lw":3,"label":"KDE"},
                            hist_kws={"histtype":"step","linewidth":"3","alpha":1,"color":"g"})
        
        return hist
    
    def hist1(self,x,y=None):
        hist1 = sns.FacetGrid(self.df,hue=y,height=5).map(sns.kdeplot,x,shade = True).add_legend()
        return hist1
    
    def corr_map(self):
        df = self.df.corr()
        plt.figure(figsize=(8.5,5.5))
        corr = sns.heatmap(df,xticklabels=df.columns,yticklabels=df.columns,annot=True)
        return corr
        
    
    
    def bar(self,x,y,z = None):
        plt.figure(figsize=(8.5,5.5))
        bruhh = sns.barplot(x = x, y = y, hue=z,data=self.df,capsize=.2)
        return bruhh
    
    def boxplot(self,x,y=None,z = None):
        box = sns.boxplot(x = x, y = y, hue=z, data=self.df,linewidth=2.5)
        return box
    
    def heatmap(self):
        pass
    
    def missingvalues_heat_map(self):
        heatmap = msno.heatmap(self.df)
        return heatmap
    
    def missingvalues_dendogram(self):
        
        dendogram = msno.dendrogram(self.df)
        return dendogram


class information():
    def __init__(self,data):
        self.df = data
    
    def info_data(self):
        """İnfo about data"""
        information = self.df.info()
        shape = self.df.shape
        return print(information,"\nBoyut Bilgisi:\n",shape)
    
    def summary_statistics(self):
        info = self.df.describe().T
        return info
    
    def value_count(self,x,y=None):
        """Write X with two quotes"""
        counts = self.df[x].value_counts()
        counts1 = self.df[y].value_counts()
        print("First Variable Value Counts:\n",counts,"\nSecond Variable Value Counts:\n",counts1)
        
    
    def nuniques(self):
        unique = self.df.nunique()
        return unique
    
    def MissingValues(self):
        missing_values = self.df.isnull().sum()
        missing_values_rate = (self.df.isnull().sum() * 100) / data.shape[0]
        df_Missing_values = missing_values
        df_Missing_values_rate = missing_values_rate
        table = pd.concat([df_Missing_values,df_Missing_values_rate],axis=1)
        table = table.rename(columns={0:"Missing_Values",1:"Missing_values_rate %"})
        return table
    
class preprocess():
    
    def __init__(self):
        print("....")
        
    
    def dummys(self):
        pass
    
    def labelencoder(self):
        pass
    
    def standardisation(self):
        pass
    
    def fillna(self):
        pass
    
    
    def outlier_describing(self):
        pass
    
    
class statistic():
    def __init__(self,df)
        self.df = df

        
    def normal(self):
        
        normal_dagilanlar = []
        normal_dagilmayanlar = []
        columns = list(df)
        alpha = 0.05
    
        for col in columns:
            result = shapiro(df[[col]])
        
            if result[1] < alpha:
                normal_dagilmayanlar.append(col)
            
            elif result[1] > alpha:
                normal_dagilanlar.append(col)
            
        
        print("Normal dagilimdan gelenler: ",normal_dagilanlar,
    
        "\nNormal dagilimdan gelmeyenler: \n",normal_dagilmayanlar)
        
        
        
    def yakala(df,Y):
        anlamli_kat_sayilar = []
        target = Y
        columns = list(df.drop(target,axis=1))
        sonuclar = []
        for col in columns:
        
            X = df[[col]]
            y = df[target]
            lm = sm.OLS(y,X)
            model = lm.fit()
            if (model.tvalues.values > 1.96):
                anlamli_kat_sayilar.append(model.tvalues.index)
        
            elif (model.tvalues.values < -1.96):
                anlamli_kat_sayilar.append(model.tvalues.index)
            
            
        for i in anlamli_kat_sayilar:
        
            print("Anlamli Kat sayilar: ",i[0])

