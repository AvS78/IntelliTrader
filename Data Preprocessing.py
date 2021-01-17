# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:35:31 2020

@author: RB
"""
# To read all the raw tick data from csv file and update
# into a combined dataframe

import pandas as pd
import numpy as np
import os

#steps - load all the monthly tick data into a list of dataframes
# then combine all the monthly data frames into a consolidated one

df={}
source_directory=r'F:\NUS FYP\Forex Data\EUR_USD_Tick'
year_directories=os.listdir(source_directory)
for year in year_directories:
    df[year]=[]
    folder_name = os.path.join(source_directory,year)
    for i in range(1,13):
        filename='DAT_ASCII_EURUSD_T_' +str(year)+str(i).zfill(2)+'.csv'
        full_file_name = os.path.join(folder_name, filename)
        df[year].append(pd.read_csv(full_file_name, names=['Timestamp','Bid','Ask'], sep=',', usecols = [0,1,2] ))


# examine if all dataframes are loaded properly by checking for shapes
for year in df:
    print("year is ", year)
    print ("length is ", len(df[year]))
    for dframe in df[year]:
        print("shape is", dframe.shape)
        
combined_df={}

for year in df:
    combined_df[year]=df[year][0]
    (row,col)=combined_df[year].shape
    total_row=row
    for i in range(1,12):
        (row,col)=df[year][i].shape
        total_row=total_row+row
        combined_df[year]=combined_df[year].append(df[year][i],ignore_index = True)
        (row,col)=combined_df[year].shape
    (row,col)=combined_df[year].shape

# check that yearly combined dataframes have data properly by looking at head and tail
for year in combined_df:
    print ('year is *******', year)
    print(combined_df[year].head())
    print(combined_df[year].tail())
    
consolidated_dataframe=pd.DataFrame()
(row,col)=consolidated_dataframe.shape
total_row=row
for year in combined_df:
    (row,col)=combined_df[year].shape
    total_row=total_row+row
    consolidated_dataframe=consolidated_dataframe.append(combined_df[year],ignore_index = True)
    (row,col)=consolidated_dataframe.shape
(row,col)=consolidated_dataframe.shape
print('combined ', 'rows= ', row)

combined_df=consolidated_dataframe

# clean raw dataframe to update the Mid column and correct format of timestamp

def update_time_stamp(dframe):
    dframe['Mid']=(dframe['Bid']+dframe['Ask'])/2
    dframe['Timestamp']=pd.to_datetime(dframe['Timestamp'].str[:8]+dframe['Timestamp'].str[9:],format='%Y%m%d%H%M%S%f')
    dframe.set_index('Timestamp', inplace=True)
    return dframe

combined_df=update_time_stamp(combined_df)
combined_df.to_pickle("combined_df.pkl")    
