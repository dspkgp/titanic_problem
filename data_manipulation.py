import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# reading data from the datasets
from read_data import read_training_data, read_test_data

pd.options.display.max_columns = 100
matplotlib.style.use('ggplot')
pd.options.display.max_rows = 100


data = read_training_data()
# reading training data

data.head()
data.describe()
# getting insights from the training data

data['Age'].fillna(data['Age'].median(), inplace=True)
# replacing all the empty entries of Age with its median
# filling the outliers
# cleaning of the data

data.describe()

survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))
plt.close()
# visualizing survival based on gender

figure = plt.figure(figsize=(13,8))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
    bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.close()
# correlating survival with age variable

figure = plt.figure(figsize=(13,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True,color = ['g','r'],
    bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.close()
# correlating ticket fair with survival

figure=plt.figure(figsize=(13,8))
ax=plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='g',s=10)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='r',s=10)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15)
plt.close()
# combining age, fare and survival on single chart

ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(13,8), ax = ax)
plt.close()
# correlating ticket fair with class

survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))
plt.close()

def status(feature):
    print 'Processing', feature, ': done'
# print function that assert whether or not a feature has been processed

# Loading the data
def get_titanic_data():
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
    titanic_data = train.append(test)
    titanic_data.reset_index(inplace=True)
    titanic_data.drop('index',inplace=True,axis=1)

    return titanic_data


def get_titles(titanic_data):
    # we extract the title from each name
    titanic_data['Title'] = titanic_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
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
    titanic_data['Title'] = titanic_data.Title.map(Title_Dictionary)

    return titanic_data

def process_age(titanic_data):
     # a function that fills the missing values of the Age variable
    titanic_data["Age"] = titanic_data.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))

    status('age')

    return titanic_data

def process_family(titanic_data):
    # introducing a new feature : the size of families (including the passenger)
    titanic_data['FamilySize'] = titanic_data['Parch'] + titanic_data['SibSp'] + 1
    # introducing other features based on the family size
    titanic_data['Singleton'] = titanic_data['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    titanic_data['SmallFamily'] = titanic_data['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    titanic_data['LargeFamily'] = titanic_data['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    status('family')

    return titanic_data

def process_ticket(titanic_data):
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'
    titanic_data['Ticket'] = titanic_data['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(titanic_data['Ticket'],prefix='Ticket')
    titanic_data = pd.concat([titanic_data, tickets_dummies],axis=1)
    titanic_data.drop('Ticket',inplace=True,axis=1)

    status('ticket')

    return titanic_data

def process_pclass(titanic_data):
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(titanic_data['Pclass'],prefix="Pclass")
    # adding dummy variables
    titanic_data = pd.concat([titanic_data,pclass_dummies],axis=1)
    # removing "Pclass"
    titanic_data.drop('Pclass',axis=1,inplace=True)
    status('pclass')

    return titanic_data

def process_sex(titanic_data):
    # mapping string values to numerical one
    titanic_data['Sex'] = titanic_data['Sex'].map({'male':1,'female':0})
    status('sex')

    return titanic_data

def process_cabin(titanic_data):
    # replacing missing cabins with U (for Unknown)
    titanic_data.Cabin.fillna('U',inplace=True)
    # mapping each Cabin value with the cabin letter
    titanic_data['Cabin'] = titanic_data['Cabin'].map(lambda c : c[0])
    # dummy encoding
    cabin_dummies = pd.get_dummies(titanic_data['Cabin'],prefix='Cabin')
    titanic_data = pd.concat([titanic_data,cabin_dummies],axis=1)
    titanic_data.drop('Cabin',axis=1,inplace=True)
    status('cabin')

    return titanic_data

def process_embarked(titanic_data):
    # two missing embarked values - filling them with the most frequent one (S)
    titanic_data.Embarked.fillna('S',inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(titanic_data['Embarked'],prefix='Embarked')
    titanic_data = pd.concat([titanic_data,embarked_dummies],axis=1)
    titanic_data.drop('Embarked',axis=1,inplace=True)

    status('embarked')

    return titanic_data

def process_name(titanic_data):
    # cleaning the name variable
    titanic_data.drop('Name',inplace=True,axis=1)
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(titanic_data['Title'],prefix='Title')
    titanic_data = pd.concat([titanic_data,titles_dummies],axis=1)
    # removing the title variables
    titanic_data.drop('Title',axis=1,inplace=True)
    status('names')

    return titanic_data

def process_fares(titanic_data):
    titanic_data.Fare.fillna(titanic_data.Fare.mean(),inplace=True)
    status('fare')

    return titanic_data

if __name__ == "__main__":
    titanic_data = get_titanic_data()
    import ipdb;ipdb.set_trace()
    titanic_data = get_titles(titanic_data)

    grouped = titanic_data.groupby(['Sex','Pclass','Title'])
    grouped.median()
    
    titanic_data = process_age(titanic_data)
    titanic_data = process_family(titanic_data)
    titanic_data = process_ticket(titanic_data)
    titanic_data = process_pclass(titanic_data)
    titanic_data = process_sex(titanic_data)
    titanic_data = process_cabin(titanic_data)
    titanic_data = process_embarked(titanic_data)
    titanic_data = process_name(titanic_data)
    titanic_data = process_fares(titanic_data)
    titanic_data.info()
    print(titanic_data.head())

	
