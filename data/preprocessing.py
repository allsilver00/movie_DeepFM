from sklearn.preprocessing import LabelEncoder
import settings
import pandas as pd




def dataProcess():
    
    data = pd.read_csv(settings.path)
    sparse_features = settings.sparse_features
    
    # 결측값 전처리
    data[sparse_features] = data[sparse_features].fillna('-1', )


    # 범주형 피처 라벨인코딩
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    return data

    
        





    
        
