# This file is for data preprocessing.

# atomic2020_DATA
# ├── train.tsv
# ├── test.tsv
# ├── dev.tsv
# └── RemoveUnderline.py (this file)

import pandas as pd
from operator import iadd

Relations = ['IsAfter', 'HasSubEvent', 'IsBefore', 'HinderedBy', 'xReason', 'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant']
Relations = [x.lower() for x in Relations]
RelationsTailThorwSubject = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xNeed', 'xIntent']
RelationsTailThorwSubject = [x.lower() for x in RelationsTailThorwSubject]
AddTo = ['oWant', 'xWant', 'xIntent']
AddTo = [x.lower() for x in AddTo]

def remove_and_save():
    '''
    Load data from .tsv, remove underline ('_'), and save data to .tsv files for train, test, dev data respectively
    train_1, test_1, dev_1 are output files
    '''

    def throw_away_subject(x):
        x = str(x)
        if x == '':
            return x
        if x.split()[0].lower() == 'personx' or x.split()[0].lower() == 'persony':
            return x[8:]
        elif x.split()[0].lower() == 'person x' or x.split()[0].lower() == 'person y':
            return x[9:]
        elif x.split()[0].lower() == 'her' or x.split()[0].lower() == 'she' or x.split()[0].lower() == 'him' or x.split()[0].lower() == 'his':
            return x[4:]
        elif x.split()[0].lower() == 'they' or x.split()[0].lower() == 'them':
            return x[5:]
        elif x.split()[0].lower() == 'he':
            return x[3:]
        elif x.split()[0].lower() == 'their':
            return x[6:]
        else:
            return x
    
    def add_to(x):
        x = str(x)
        if x.split()[0].lower() == 'to':
            return x
        else:
            return 'to ' + x

    df_train = pd.read_csv('train.tsv', sep='\t')
    df_test = pd.read_csv('test.tsv', sep='\t')
    df_dev = pd.read_csv('dev.tsv', sep='\t')

    # processing train data

    df_train['head_event_contain_under']=df_train['head_event'].apply(lambda x: '_' in x)    
    df_train['tail_event_contain_none'] = df_train['tail_event'].apply(lambda x: 'none' in str(x).lower() )   
    train_data=df_train[df_train.head_event_contain_under==False]
    train_data=train_data[train_data.tail_event_contain_none == False]
    
    train_data['stay'] = train_data['relation'].apply(lambda x: x.lower() in Relations)    
    train_data = train_data[train_data.stay == True]
    train_data = train_data.drop(columns = ['stay', 'head_event_contain_under', 'tail_event_contain_none'])

    # processing has subevent
    train_data['HSE'] = train_data['relation'].apply(lambda x: 'HasSubEvent'.lower() == x.lower())   
    train_data.loc[train_data['HSE'] == True, 'head_event'] = train_data[train_data['HSE'] == True]['head_event'].apply(lambda x: 'PersonX ' + x)
    train_data.loc[train_data['HSE'] == True, 'tail_event'] = train_data[train_data['HSE'] == True]['tail_event'].apply(lambda x: ': PersonX ' + x)
    train_data = train_data.drop(columns=['HSE'])

    # add personX to tail of Xreason
    train_data['reason'] = train_data['relation'].apply(lambda x: 'xreason'.lower() == x.lower())   
    train_data.loc[train_data['reason'] == True, 'head_event'] = train_data[train_data['reason'] == True]['head_event'].apply(lambda x: 'PersonX ' + x)
    train_data.loc[train_data['reason'] == True, 'tail_event'] = train_data[train_data['reason'] == True]['tail_event'].apply(lambda x: 'PersonX ' + x)
    train_data = train_data.drop(columns=['reason'])

    # throw some subjective
    train_data['throw_subjective'] = train_data['relation'].apply(lambda x: x.lower() in RelationsTailThorwSubject) 
    train_data.loc[train_data['throw_subjective'] == True, 'tail_event'] = train_data[train_data['throw_subjective'] == True]['tail_event'].apply(throw_away_subject)
    train_data = train_data.drop(columns=['throw_subjective'])

    # add 'to' to 'xWant', 'oWant', 'xIntend'
    train_data['add_to'] = train_data['relation'].apply(lambda x: x.lower() in AddTo) 
    train_data.loc[train_data['add_to'] == True, 'tail_event'] = train_data[train_data['add_to'] == True]['tail_event'].apply(add_to)
    train_data = train_data.drop(columns=['add_to'])

    # processing test data

    df_test['head_event_contain_under']=df_test['head_event'].apply(lambda x: '_' in x)    
    df_test['tail_event_contain_none'] = df_test['tail_event'].apply(lambda x: 'none' in str(x).lower() )   
    test_data=df_test[df_test.head_event_contain_under==False]
    test_data=test_data[test_data.tail_event_contain_none == False]
    test_data['stay'] = test_data['relation'].apply(lambda x: x.lower() in Relations)    
    test_data = test_data[test_data.stay == True]
    test_data = test_data.drop(columns = ['stay', 'head_event_contain_under', 'tail_event_contain_none'])

    test_data['HSE'] = test_data['relation'].apply(lambda x: 'HasSubEvent'.lower() == x.lower())   
    test_data.loc[test_data['HSE'] == True, 'head_event'] = test_data[test_data['HSE'] == True]['head_event'].apply(lambda x: 'PersonX ' + x)
    test_data.loc[test_data['HSE'] == True, 'tail_event'] = test_data[test_data['HSE'] == True]['tail_event'].apply(lambda x: ': PersonX ' + x)
    test_data = test_data.drop(columns=['HSE'])

    test_data['reason'] = test_data['relation'].apply(lambda x: 'xreason'.lower() == x.lower())   
    test_data.loc[test_data['reason'] == True, 'head_event'] = test_data[test_data['reason'] == True]['head_event'].apply(lambda x: 'PersonX ' + x)
    test_data.loc[test_data['reason'] == True, 'tail_event'] = test_data[test_data['reason'] == True]['tail_event'].apply(lambda x: 'PersonX ' + x)
    test_data = test_data.drop(columns=['reason'])

    test_data['throw_subjective'] = test_data['relation'].apply(lambda x: x.lower() in RelationsTailThorwSubject) 
    test_data.loc[test_data['throw_subjective'] == True, 'tail_event'] = test_data[test_data['throw_subjective'] == True]['tail_event'].apply(throw_away_subject)
    test_data = test_data.drop(columns=['throw_subjective'])

    test_data['add_to'] = test_data['relation'].apply(lambda x: x.lower() in AddTo) 
    test_data.loc[test_data['add_to'] == True, 'tail_event'] = test_data[test_data['add_to'] == True]['tail_event'].apply(add_to)
    test_data = test_data.drop(columns=['add_to'])

    # processing val data

    df_dev['head_event_contain_under']=df_dev['head_event'].apply(lambda x: '_' in x)    
    df_dev['tail_event_contain_none'] = df_dev['tail_event'].apply(lambda x: 'none' in str(x).lower() )   
    dev_data=df_dev[df_dev.head_event_contain_under==False]
    dev_data=dev_data[dev_data.tail_event_contain_none == False]
    dev_data['stay'] = dev_data['relation'].apply(lambda x: x.lower() in Relations)    
    dev_data = dev_data[dev_data.stay == True]
    dev_data = dev_data.drop(columns = ['stay', 'head_event_contain_under', 'tail_event_contain_none'])

    dev_data['HSE'] = dev_data['relation'].apply(lambda x: 'HasSubEvent'.lower() == x.lower())   
    dev_data.loc[dev_data['HSE'] == True, 'head_event'] = dev_data[dev_data['HSE'] == True]['head_event'].apply(lambda x: 'PersonX ' + x)
    dev_data.loc[dev_data['HSE'] == True, 'tail_event'] = dev_data[dev_data['HSE'] == True]['tail_event'].apply(lambda x: ': PersonX ' + x)
    dev_data = dev_data.drop(columns=['HSE'])

    dev_data['reason'] = dev_data['relation'].apply(lambda x: 'xreason'.lower() == x.lower())   
    dev_data.loc[dev_data['reason'] == True, 'head_event'] = dev_data[dev_data['reason'] == True]['head_event'].apply(lambda x: 'PersonX ' + x)
    dev_data.loc[dev_data['reason'] == True, 'tail_event'] = dev_data[dev_data['reason'] == True]['tail_event'].apply(lambda x: 'PersonX ' + x)
    dev_data = dev_data.drop(columns=['reason'])

    dev_data['throw_subjective'] = dev_data['relation'].apply(lambda x: x.lower() in RelationsTailThorwSubject) 
    dev_data.loc[dev_data['throw_subjective'] == True, 'tail_event'] = dev_data[dev_data['throw_subjective'] == True]['tail_event'].apply(throw_away_subject)
    dev_data = dev_data.drop(columns=['throw_subjective'])

    dev_data['add_to'] = dev_data['relation'].apply(lambda x: x.lower() in AddTo) 
    dev_data.loc[dev_data['add_to'] == True, 'tail_event'] = dev_data[dev_data['add_to'] == True]['tail_event'].apply(add_to)
    dev_data = dev_data.drop(columns=['add_to'])


    # save to tsv

    train_data.to_csv('train_1.tsv', sep="\t", index=False)
    test_data.to_csv('test_1.tsv', sep="\t", index=False)
    dev_data.to_csv('dev_1.tsv', sep="\t", index=False)

    print("num of training data: ", train_data.shape[0] - 1)
    print("num of testing data: ", test_data.shape[0] - 1)
    print("num of dev data: ", dev_data.shape[0] - 1)


if __name__ == '__main__':
    remove_and_save()