#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:33:29 2023

@author: bohlen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/data/hu_bohlen/Documents/limesurvey_results/results-survey25-05.csv')

# View data
#print(data)

# Check the column names
#print(data.columns)

# clean data, only of the probands, no pilots
data_clean = data.iloc[7:, :]

# the uncertainty questionnaire
questionnaire_data = data_clean.iloc[:, 8:26]

# single questions
question_name = '1. Ich kann mich nicht entspannen, wenn ich nicht weiß, was morgen passieren wird. '

question_data = questionnaire_data[question_name]

#plot of the answers to a single question (1 not at all characteristic of me; 5 entirely characteristic of me)
question_data.plot(kind='bar')

#scale of x-/ y-axis
x_ticks = [0, 1, 2, 3, 4]
x_labels = ['001', '002', '003', '004', '005']
plt.xticks(x_ticks, x_labels)

y_ticks = [0, 1, 2, 3, 4, 5]
y_labels = ['0', '1', '2', '3', '4', '5']
plt.yticks(y_ticks, y_labels)

#label of the axis
plt.xlabel('ID')
plt.ylabel('Response')
plt.title('Ich kann mich nicht entspannen, wenn ich nicht weiß, was morgen passieren wird.')
plt.show()


# calculate the mean for all questions
questionnaire_data.mean()

questionnaire_data_means = questionnaire_data.mean()

#plot the questionnaire mean data
# 1 not at all characteristic of me; 5 entirely characteristic of me
questionnaire_data_means.plot(kind='bar')
plt.xlabel('Questions')
plt.ylabel('Mean')
plt.title('Mean of Responses for Each Question')
plt.show()

#general sum of the uncertainty questionnaire
#create a loop to calculate the sum for every proband
# Convert the list to a NumPy array
questionnaire_data_array = np.array(questionnaire_data)

# Initialize an empty list to store row (proband) sums
row_sums = []

# Iterate over the rows of the array
for row in questionnaire_data_array:
    # Perform operations on each row
    paticipant_mean = row.mean()
    print(paticipant_mean)
        
for row in questionnaire_data_array:
    paticipant_sum = row.sum()
    #to add single rows to the list row_sums
    row_sums.append(paticipant_sum)
    print(paticipant_sum)

#plot the sum of each proband in a bar plot
plt.bar(range(len(row_sums)), row_sums)
x_ticks = [0, 1, 2, 3, 4]
x_labels = ['001', '002', '003', '004', '005']
plt.xticks(x_ticks, x_labels)
plt.xlabel('Proband')
plt.ylabel('Sum')
plt.title('Sum of the probands')
plt.show()

#plot the sum of each proband in a boxplot
plt.boxplot(row_sums)
plt.title('Boxplot of sums of the uncertainty questionnaire')
plt.xlabel('Probands')
plt.xticks([1], [''])
plt.ylabel('Sums of the uncertainty questionnaire')
plt.show()

#calculate the sums for the different subgroups
#Eingeschränkte Handlungsfähigkeit (7, 8, 9, 10, 13, 15) -> eh_data (14, 15, 16, 17, 20, 22)
#Belastung (1, 2, 3, 16, 17, 18)                         -> b_data  (8, 9, 10, 23, 24, 25)
#Vigilanz (4, 5, 6, 11, 12, 14)                          -> v_data  (11, 12, 13, 18, 19, 21)

#create datasets
eh_data = data_clean.iloc[:, [14, 15, 16, 17, 20, 22]]
b_data = data_clean.iloc[:, [8, 9, 10, 23, 24, 25]]
v_data = data_clean.iloc[:, [11, 12, 13, 18, 19, 21]]

#Eingeschränkte Handlungsfähigkeit
#create a loop to calculate the sum for every proband
# Convert the list to a NumPy array
eh_data_array = np.array(eh_data)

# Initialize an empty list to store row (proband) sums
eh_sums = []

for row in eh_data_array:
    eh_paticipant_sum = row.sum()
    #to add single rows to the list eh_sums
    eh_sums.append(eh_paticipant_sum)
    print(eh_paticipant_sum)
    
#plot the sum of each proband in a bar plot
plt.bar(range(len(eh_sums)), eh_sums)
x_ticks = [0, 1, 2, 3, 4]
x_labels = ['001', '002', '003', '004', '005']
plt.xticks(x_ticks, x_labels)
plt.xlabel('Proband')
plt.ylabel('Sum')
plt.title('Sum of the probands for Eingeschränkte Handlungsfähigkeit')
plt.show()

#plot the sum of each proband in a boxplot
plt.boxplot(eh_sums)
plt.title('Boxplot of sums of the uncertainty questionnaire (Eingeschränkte Handlungsfähigkeit)')
plt.xlabel('Probands')
plt.xticks([1], [''])
plt.ylabel('Sums of the uncertainty questionnaire (Eingeschränkte Handlungsfähigkeit)')
plt.show()   

 #Belastung
b_data_array = np.array(b_data)

# Initialize an empty list to store row (proband) sums
b_sums = []

for row in b_data_array:
    b_paticipant_sum = row.sum()
    #to add single rows to the list eh_sums
    b_sums.append(b_paticipant_sum)
    print(b_paticipant_sum)
    
#plot the sum of each proband in a bar plot
plt.bar(range(len(b_sums)), b_sums)
x_ticks = [0, 1, 2, 3, 4]
x_labels = ['001', '002', '003', '004', '005']
plt.xticks(x_ticks, x_labels)
plt.xlabel('Proband')
plt.ylabel('Sum')
plt.title('Sum of the probands for Belastung')
plt.show()

#plot the sum of each proband in a boxplot
plt.boxplot(b_sums)
plt.title('Boxplot of sums of the uncertainty questionnaire (Belastung)')
plt.xlabel('Probands')
plt.xticks([1], [''])
plt.ylabel('Sums of the uncertainty questionnaire (Belastung)')
plt.show() 

#Vigilanz
v_data_array = np.array(v_data)

# Initialize an empty list to store row (proband) sums
v_sums = []

for row in v_data_array:
    v_paticipant_sum = row.sum()
    #to add single rows to the list eh_sums
    v_sums.append(v_paticipant_sum)
    print(v_paticipant_sum)
    
#plot the sum of each proband in a bar plot
plt.bar(range(len(v_sums)), v_sums)
x_ticks = [0, 1, 2, 3, 4]
x_labels = ['001', '002', '003', '004', '005']
plt.xticks(x_ticks, x_labels)
plt.xlabel('Proband')
plt.ylabel('Sum')
plt.title('Sum of the probands for Vigilanz')
plt.show()

#plot the sum of each proband in a boxplot
plt.boxplot(v_sums)
plt.title('Boxplot of sums of the uncertainty questionnaire (Vigilanz)')
plt.xlabel('Probands')
plt.xticks([1], [''])
plt.ylabel('Sums of the uncertainty questionnaire (Vigilanz)')
plt.show() 
