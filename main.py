
from data.preprocessing import dataProcess
from model.DeepFM_model import run_DeepFM
import settings
from data.result import pred_result,save

if __name__=="__main__":

    # 데이터 라벨인코딩
    data = dataProcess()
    print('data_labelEncoder')

    # 모델 학습, 평가
    sparse_features = settings.sparse_features
    target = settings.target
    
    feature_names, model = run_DeepFM(data, sparse_features, target)
    print("model evaluation")


    # 예측결과 확인 
    pred_result(data, feature_names, model)

    # 예측결과 디코딩하여 저장
    result_data = pred_result(data, feature_names, model)

    save(result_data)

 