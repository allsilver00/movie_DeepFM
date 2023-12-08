
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import settings

def pred_result(data,feature_names,model):

    predict_model = {name: data[name] for name in feature_names}
    data['pred'] = model.predict(predict_model, batch_size=256)

    return data


def save(data):

    # 영화이름 디코딩 전
    result_df = data[['userId', 'title', 'pred']]
    copy_df = pd.read_csv(settings.path)
    # 영화이름 디코딩 후 파일 저장

    lbe = LabelEncoder()
    lbe.fit(copy_df['title'])
    result_df['title'] = lbe.inverse_transform(result_df['title'])
    result_df.to_csv('decoded_pre.csv', index=False)
  
