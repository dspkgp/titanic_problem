import pandas as pd
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt

# reading data from the datasets
from read_data import read_training_data, read_test_data, read_gender_data

pd.options.display.max_columns = 100
matplotlib.style.use('ggplot')
pd.options.display.max_rows = 100

# reading training data
data = read_training_data()

# getting insights from the training data
data.head()
data.describe()

# replacing all the empty entries of Age with its median
data['Age'].fillna(data['Age'].median(), inplace=True)

data.describe()

# visualizing survival based on gender
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))
plt.close()

# correlating survival with age variable
figure = plt.figure(figsize=(13,8))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
    bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.close()

# correlating ticket fair with survival
figure = plt.figure(figsize=(13,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True,color = ['g','r'],
    bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.close()

# combining age, fare and survival on single chart
figure=plt.figure(figsize=(13,8))
ax=plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='g',s=10)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='r',s=10)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15)
plt.close()

# correlating ticket fair with class
ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(13,8), ax = ax)
plt.close()

survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))
plt.close()

# print function that assert whether or not a feature has been processed
def status(feature):
    print('Processing', feature, ': done')

# Loading the data
def get_combined_data():
    # reading train data
    train = read_training_data()
    # reading test data
    test = read_test_data()
    # import ipdb; ipdb.set_trace()
    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    # merging train data and test data for future feature engineering
    # import ipdb; ipdb.set_trace()
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)

    return combined

def get_titles():

    global combined
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    # a map of more aggregated titles
    Title_Dictionary = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"

    }
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)

def process_age():
    global combined
     # a function that fills the missing values of the Age variable
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())

    status('age')

def process_family():
    global combined

    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    status('family')

def process_ticket():

    global combined
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')

def process_pclass():
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    status('pclass')

def process_sex():
    global combined
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    status('sex')

def process_cabin():
    global combined
    # replacing missing cabins with U (for Unknown)
    combined.Cabin.fillna('U',inplace=True)
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    # dummy encoding
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    combined = pd.concat([combined,cabin_dummies],axis=1)
    combined.drop('Cabin',axis=1,inplace=True)
    status('cabin')

def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)

    status('embarked')

def process_name():
    global combined
    # cleaning the name variable
    combined.drop('Name',inplace=True,axis=1)
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    # removing the title variables
    combined.drop('Title',axis=1,inplace=True)
    status('names')

def process_fares():
    global combined
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)
    status('fare')


def main():

    global combined
    combined = get_combined_data()
    get_titles()
    process_age()
    process_family()
    process_ticket()
    process_pclass()
    process_sex()
    process_cabin()
    process_embarked()
    process_name()
    process_fares()
    return combined

main()
