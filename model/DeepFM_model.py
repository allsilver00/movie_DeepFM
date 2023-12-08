
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.feature_column import SparseFeat,get_feature_names
from deepctr.models import DeepFM





def run_DeepFM (data, sparse_features, target):

    # 모델에 input 데이터 생성
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=16, use_hash=False, dtype='int32'
                ) for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


    train, test = train_test_split(data, test_size=0.2, random_state=2020)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 모델 정의, 훈련, 예측, 평가
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                metrics=['binary_crossentropy','AUC'] )

    history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2 )
    
    pred_ans = model.predict(test_model_input, batch_size=256)

    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    return feature_names, model







