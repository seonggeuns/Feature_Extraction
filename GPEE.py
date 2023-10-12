from sklearn.model_selection import train_test_split
from FeatureExtraction import feature_extraction
import copy
def GPFE(data, test_split_portion: float,kfold,ML_model,GP_config: dict, unit: dict):

    data1=copy.deepcopy(data)
    train_unit = []
    for feature in unit.keys():
        if unit[feature] == None:
            data1 = data1.drop([feature], axis=1)
        else:
            train_unit.append(unit[feature])

    attribute_name = [column for column in data1.columns[:-1]]
    attribute = dict(zip(attribute_name, train_unit))


    if test_split_portion:
        train, test = train_test_split(data, stratify=data['Decision'], test_size=test_split_portion, random_state=2)
        train,test=train.to_dict(),test.to_dict()

    else:
        train, test = train_test_split(data, stratify=data['Decision'], test_size=0.3, random_state=2)
        train,test=train.to_dict(),test.to_dict()



    data = data.to_dict()

    feature_extraction(GP_config, ML_model, attribute, attribute_name, train,test, data, train_unit,kfold)


    # 여기서 train,test로만 나누고,,,, -> 그 다음에 feature extraction에서 fold로 나누기?

    # k-fold 기능 추가안하면 그냥 train accuracy 구하는거,,, 만약 kfold 추가하면..?

    # validation_split_portion None이라면 validation -> training
    # fold로 분리 후 FE 적용 or FE 적용 후 fold 분리? -> 사실상 똑같음




