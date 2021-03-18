import numpy as np 
import pandas as pd 
from datetime import datetime
import math
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import gc
from sklearn.metrics import roc_auc_score


def tryconvert(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def clean_tours(value):
    try:
        return (value-datetime(2012,1,1)).days
    except (OutOfBoundsDatetime):
        return 0


def common_elements(list1, list2):
    return [element for element in list1 if element in list2]


train = pd.read_csv(r'../../data/train.csv')
test = pd.read_csv(r'../../data/test.csv')
bikers = pd.read_csv(r'../../data/bikers.csv')
bike_net = pd.read_csv(r'../../data/bikers_network.csv')
bike_net['friends'] = bike_net['friends'].str.split().tolist()
bike_net['friends_len'] = bike_net['friends'].apply(lambda x:0 if x!= x else len(x))
tours = pd.read_csv(r'../../data/tours.csv')
tour_conv = pd.read_csv(r'../../data/tour_convoy.csv')
tour_conv['going'] = tour_conv['going'].str.split().tolist()
tour_conv['maybe'] = tour_conv['maybe'].str.split().tolist()
tour_conv['invited'] = tour_conv['invited'].str.split().tolist()
tour_conv['not_going'] = tour_conv['not_going'].str.split().tolist()
columns = ['going','maybe','invited','not_going']
for c in columns:
    tour_conv[c+'_len'] = tour_conv[c].apply(lambda x:0 if x!= x else len(x))

tours['tour_date'] = tours['tour_date'].apply(lambda x:datetime.strptime(x, '%d-%m-%Y'))
test['timestamp'] = test['timestamp'].apply(lambda x:datetime.strptime(x, '%d-%m-%Y %H:%M:%S'))
train['timestamp'] = train['timestamp'].apply(lambda x:datetime.strptime(x, '%d-%m-%Y %H:%M:%S'))
test['hour'] = (pd.DatetimeIndex(test['timestamp'])).hour
train['hour'] = (pd.DatetimeIndex(train['timestamp'])).hour
test['day'] = (pd.DatetimeIndex(test['timestamp'])).day
train['day'] = (pd.DatetimeIndex(train['timestamp'])).day
test['month'] = (pd.DatetimeIndex(test['timestamp'])).month
train['month'] = (pd.DatetimeIndex(train['timestamp'])).month
test['year'] = (pd.DatetimeIndex(test['timestamp'])).year
train['year'] = (pd.DatetimeIndex(train['timestamp'])).year
test['day_of_week'] = (pd.DatetimeIndex(test['timestamp'])).dayofweek
train['day_of_week'] = (pd.DatetimeIndex(train['timestamp'])).dayofweek
tours['tour_date'] = tours['tour_date'].apply(lambda x: clean_tours(x))

test.drop(columns=['timestamp'],inplace = True)
train.drop(columns=['timestamp'],inplace = True)

pos_given = []
dist = []
for lon,lat in zip(tours['latitude'],tours['longitude']):
    if pd.isnull(lon) or pd.isnull(lat):
        pos_given.append(0)
        dist.append(0)
    else:
        pos_given.append(1)
        dist.append(math.sqrt(math.pow(lon,2)+math.pow(lat,2)))

tours.drop(columns=['city','state','pincode','country'],inplace = True)
tours['dist'] = dist
tours['pos_given'] = pos_given
tours['latitude'].fillna(0,inplace=True)
tours['longitude'].fillna(1993,inplace=True)

bikers['member_since'] = bikers['member_since'].apply(lambda x:datetime.strptime(x, '%d-%m-%Y') if (not pd.isnull(x) and x != '--None') else None)
bikers['member_since'].fillna(method='ffill',inplace=True)
bikers['bornIn'] = bikers['bornIn'].apply(lambda x: tryconvert(x))
bikers['bornIn'].fillna(1993,inplace=True)
bikers['time_zone'].fillna(420,inplace=True)
bikers['gender'].fillna('male',inplace=True)
bikers.drop(columns=['language_id'],inplace=True)
bikers['location_id'] = bikers['location_id'].astype('category')
bikers['location_id'] = bikers['location_id'].cat.codes
bikers['gender'] = bikers['gender'].astype('category')
bikers['gender'] = bikers['gender'].cat.codes
bikers['month_joined'] = (pd.DatetimeIndex(bikers['member_since']) - datetime(2012,1,1)).days
bikers.drop(columns=['area','member_since'],inplace=True)

trainbt = pd.merge(pd.merge(train, bikers, on = "biker_id", how = "inner"), tours, on = "tour_id", how = "inner") 
trainbt['biker_id'] = trainbt['biker_id_x']
trainbt = trainbt.drop('biker_id_x', axis=1)
final_train = pd.merge(pd.merge(trainbt, bike_net, on = "biker_id", how = "inner") , tour_conv, on = "tour_id", how = "inner") 

testbt = pd.merge(pd.merge(test, bikers, on = "biker_id", how = "inner"), tours, on = "tour_id", how = "inner") 
testbt['biker_id'] = testbt['biker_id_x']
testbt = testbt.drop('biker_id_x', axis=1)
final_test = pd.merge(pd.merge(testbt, bike_net, on = "biker_id", how = "inner"), tour_conv, on = "tour_id", how = "inner") 

friends_going = []
for i in range(len(final_train)):
    list1 = final_train['friends'].loc[i]
    list2 = final_train['going'].loc[i]    
    if list1!=list1 or list2!=list2:
        friends_going.append(0)
    else:
        friends_going.append(len(common_elements(list1,list2)))
friends_not_going = []
for i in range(len(final_train)):
    list1 = final_train['friends'].loc[i]
    list2 = final_train['not_going'].loc[i]    
    if list1!=list1 or list2!=list2:
        friends_not_going.append(0)
    else:
        friends_not_going.append(len(common_elements(list1,list2)))
friends_maybe = []
for i in range(len(final_train)):
    list1 = final_train['friends'].loc[i]
    list2 = final_train['maybe'].loc[i]    
    if list1!=list1 or list2!=list2:
        friends_maybe.append(0)
    else:
        friends_maybe.append(len(common_elements(list1,list2)))
final_train['friends_going'] = friends_going
final_train['friends_not_going'] = friends_going
final_train['friends_maybe'] = friends_going

friends_going = []
for i in range(len(final_test)):
    list1 = final_test['friends'].loc[i]
    list2 = final_test['going'].loc[i]    
    if list1!=list1 or list2!=list2:
        friends_going.append(0)
    else:
        friends_going.append(len(common_elements(list1,list2)))
friends_not_going = []
for i in range(len(final_test)):
    list1 = final_test['friends'].loc[i]
    list2 = final_test['not_going'].loc[i]    
    if list1!=list1 or list2!=list2:
        friends_not_going.append(0)
    else:
        friends_not_going.append(len(common_elements(list1,list2)))
friends_maybe = []
for i in range(len(final_test)):
    list1 = final_test['friends'].loc[i]
    list2 = final_test['maybe'].loc[i]    
    if list1!=list1 or list2!=list2:
        friends_maybe.append(0)
    else:
        friends_maybe.append(len(common_elements(list1,list2)))
final_test['friends_going'] = friends_going
final_test['friends_not_going'] = friends_going
final_test['friends_maybe'] = friends_going

final_train.drop(columns=['friends','going','maybe','invited_y','not_going','dislike'],inplace=True)
final_test.drop(columns=['friends','going','maybe','invited_y','not_going'],inplace=True)

cols = list(final_train.keys())
cols.remove('like')
cols.remove('tour_id')
cols.remove('biker_id')
cols.remove('biker_id_y')
X = final_train[cols]
y = final_train['like']

folds = KFold(n_splits=10, shuffle=True, random_state=123)
oof_preds = np.zeros(final_train.shape[0])
sub_preds = np.zeros(final_test.shape[0])
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    trn_x, trn_y = X.iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]
    print(trn_x.shape, trn_y.shape, val_x.shape, val_y.shape)
    clf = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=100,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=20,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=500, early_stopping_rounds=400
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(final_test[cols], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()

final_test['Target'] = sub_preds
final_test['Target'] = final_test['Target']
bikers_df = final_test.drop_duplicates(subset="biker_id")
bikers_set = np.array(bikers_df["biker_id"])
bikers = []
tours = []
for biker in bikers_set:
    subset = final_test.loc[final_test['biker_id']==biker,['biker_id','tour_id','Target']]
    subset = subset.sort_values(by='Target', ascending=False).reset_index(drop=True)
    tour_list = list(subset['tour_id'])
    tour = " ".join(tour_list)
    bikers.append(biker)
    tours.append(tour)
d = {"biker_id":bikers,"tour_id":tours}
submission1 =pd.DataFrame(d,columns=["biker_id","tour_id"])
submission1.to_csv(r"../../data/ME17B027_NA17B008_1.csv",index=False)

final_train.drop(columns=['hour'],inplace=True)
final_test.drop(columns=['hour'],inplace=True)
cols.remove('hour')
X = final_train[cols]
y = final_train['like']

folds = KFold(n_splits=10, shuffle=True, random_state=123)
oof_preds = np.zeros(final_train.shape[0])
sub_preds = np.zeros(final_test.shape[0])
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    trn_x, trn_y = X.iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]
    print(trn_x.shape, trn_y.shape, val_x.shape, val_y.shape)
    clf = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=100,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=20,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=500, early_stopping_rounds=400
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(final_test[cols], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()

final_test['Target'] = sub_preds
final_test['Target'] = final_test['Target']
bikers_df = final_test.drop_duplicates(subset="biker_id")
bikers_set = np.array(bikers_df["biker_id"])
bikers = []
tours = []
for biker in bikers_set:
    subset = final_test.loc[final_test['biker_id']==biker,['biker_id','tour_id','Target']]
    subset = subset.sort_values(by='Target', ascending=False).reset_index(drop=True)
    tour_list = list(subset['tour_id'])
    tour = " ".join(tour_list)
    bikers.append(biker)
    tours.append(tour)
d = {"biker_id":bikers,"tour_id":tours}
submission2 =pd.DataFrame(d,columns=["biker_id","tour_id"])
submission2.to_csv(r"../../data/ME17B027_NA17B008_2.csv",index=False)
