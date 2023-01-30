import pandas as pd
import numpy as np
import argparse
from random_forest import RandomForest

def parse_args():
    parser = argparse.ArgumentParser(description='Run random forrest with specified input arguments')
    parser.add_argument('--n-classifiers', type=int,
                        help='number of features to use in a tree',
                        default=1)
    parser.add_argument('--train-data', type=str, default='data.csv',
                        help='train data path')
    parser.add_argument('--test-data', type=str, default='data.csv',
                        help='test data path')
    parser.add_argument('--criterion', type=str, default='entropy',
                        help='criterion to use to split nodes. Should be either gini or entropy.')
    parser.add_argument('--maxdepth', type=int, help='maximum depth of the tree',
                        default=5)
    parser.add_argument('--min-sample-split', type=int, help='The minimum number of samples required to be at a leaf node',
                        default=20)
    parser.add_argument('--max-features', type=int,
                        help='number of features to use in a tree',
                        default=12)
    a = parser.parse_args()

    return(a.n_classifiers, a.train_data, a.test_data, a.criterion, a.maxdepth, a.min_sample_split, a.max_features)


def read_data(path):
    data = pd.read_csv(path)
    return data

def data_preprocessing(data):
    #remove White space from catagorical data
    for i in ["workclass", "education", "marital-status", "occupation", "race" , "sex", "native-country"]:
        data[i]=data[i].str.strip()

    #turn all ? to nan
    data = data.replace("?", np.nan)

    #fill missing data with their mode
    cols = ["workclass", "native-country", "occupation"]
    data[cols]=data[cols].fillna(data.mode().iloc[0])

    #fnlwgt does not have perdictive power and education is juat a label of education_num, so I drop them
    data.drop(["fnlwgt","education"], axis=1, inplace = True)

    #Combine capital_gain and capital-loss to one features
    data['capital'] = data.apply(lambda row: row["capital-gain"] - row["capital-loss"], axis=1)
    data.drop(["capital-gain","capital-loss"], axis=1, inplace = True)

    #there are 41 countries, 90% of data are from US, I will split data into "US" and "Others", so we dont have to create 41 new node in split
    data['native-country'] = data.apply(lambda row: 'United-States' if row["native-country"] == "United-States"  else 'Other', axis=1)

    #for somereason income in testing dataset have a dot
    data = data.replace("<=50K.","<=50K") 
    data = data.replace(">50K.",">50K")

    return data


def main():
    n_classifiers, train_data_path, test_data_path, criterion, max_depth, min_sample_split, max_features = parse_args()
    train_data = read_data(train_data_path)
    test_data = read_data(test_data_path)

    
    # YOU NEED TO HANDLE MISSING VALUES HERE
    train_data= data_preprocessing(train_data)
    test_data = data_preprocessing(test_data)
    random_forest = RandomForest(n_classifiers=n_classifiers,
                  criterion = criterion,
                  max_depth=  max_depth,
                  min_samples_split = min_sample_split ,
                  max_features = max_features )

    print(f"Accuracy of Training validation using {criterion}: {random_forest.fit(train_data, 'income')}")
    #print(random_forest.evaluate(train_data, 'income'))
    print(f"Accuracy of Testing validation using {criterion}: {random_forest.evaluate(test_data, 'income')}")


if __name__ == '__main__':
    main()

