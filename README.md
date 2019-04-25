# Machine-Learning

0412--------------------------------------------------------------------------
-- FirstModel(48x48x1)-----------------------
learning_rate: 0.000300
training_epochs: 150
batch_size: 4000
Cost: 0.012018
Accuracy: 1.000000 
Comment: "데이터가 부족하여 traing data 와 test data 를 같게 함(정확도 의미 없음)."

0415--------------------------------------------------------------------------
train 1063 pieces
test 108 pieces (20, 51, 7, 8, 13, 3, 6)
Comment: "학습데이터의 약 10%를 검증 데이터로 사용."

-- FirstModel(48x48x1)-----------------------
learning_rate: 0.000300
training_epochs: 150
batch_size: 500
Cost: 0.000000
Accuracy: 0.361111
Comment: "과적합 

-- FirstModel(48x48x1)-----------------------
learning_rate: 0.000300
training_epochs: 85
batch_size: 500
Cost: 0.000473
Accuracy: 0.444444 ↑
Comment: "과적합 제거 후 정확도 향상."

0425--------------------------------------------------------------------------
Comment: "4월 25일부터 OpenCV로 얼굴만 자른 사진을 학습데이터로 이용 시작!"
OpenCV preprocessing!! 
train 1668 (245, 809, 95, 124, 250, 42, 103)
test 181 (25, 90, 10, 13, 28, 4, 11)

-- FirstModel(48x48x1)-----------------------
learning_rate: 0.000300
training_epochs: 85
batch_size: 500
Cost: 0.000226
Accuracy: 0.662983 ↑
face 0 miss classification: 10, accuracy: 0.600000
face 1 miss classification: 2, accuracy: 0.977778
face 2 miss classification: 8, accuracy: 0.200000
face 3 miss classification: 12, accuracy: 0.076923
face 4 miss classification: 20, accuracy: 0.285714
face 5 miss classification: 2, accuracy: 0.500000
face 6 miss classification: 7, accuracy: 0.363636
Comment: " OpenCV 전처리 과정 후 정확도가 약 22% 향상.
표정별 오분류 개수를 카운트하여, 표정별 분류 정확도를 세분화 함. 
학습데이터의 양에 비례하여 정확도가 올라가는 것을 확인 할 수 있음.
5번 표정은 학습, 검증 데이터 수가 너무 낮으므로 정확도를 신뢰할 수 없음."

-- FirstModel(96x96x1)-----------------------
learning_rate: 0.000300
training_epochs: 85
batch_size: 150
Cost: 0.000001
Accuracy: 0.668508 ↑
face 0 miss classification: 6, accuracy: 0.760000 ↑
face 1 miss classification: 10, accuracy: 0.888889 ↓
face 2 miss classification: 8, accuracy: 0.200000 =
face 3 miss classification: 11, accuracy: 0.153846 ↑
face 4 miss classification: 13, accuracy: 0.535714 ↑
face 5 miss classification: 3, accuracy: 0.250000 ↓
face 6 miss classification: 9, accuracy: 0.181818 ↓
Comment: "학습 데이터 사이즈를 48x48x1에서 96x96x1로 확대하여 학습해 본 결과 정확도가 약 1% 향상.
표정 1번은 학습데이터 수가 굉장히 많음에도 불구하고 사진을 96x96까지 확대하여 과적합 된 것으로 예상...
표정 5번, 6번은 두려움, 역겨움으로 96x96으로 확대 후 정확도는 떨어졌으나, 
표정분류 기준이 늘어난 것에 비해 데이터 량이 부족한 것이 원인인 것으로 예상."

-- FirstModel(96x96x1)-----------------------
learning_rate: 0.000300
training_epochs: 50
batch_size: 150
Cost: 0.000009
Accuracy: 0.629834 ↓
face 0 miss classification: 20, accuracy: 0.200000 ↓
face 1 miss classification: 8, accuracy: 0.911111 ↑
face 2 miss classification: 10, accuracy: 0.000000 ↓
face 3 miss classification: 11, accuracy: 0.153846 = 
face 4 miss classification: 8, accuracy: 0.714286 ↑
face 5 miss classification: 3, accuracy: 0.250000 =
face 6 miss classification: 7, accuracy: 0.363636 ↑
Comment: "과적합 우려로 학습을 좀 적게 시켜보았으나 전체적인 정확도 하락.
0번, 2번 표정은 정확도가 급격히 떨어지고,
1번(과적합 제거로 정확도 향상), 4번, 6번 표정은 정확도 향상."

-- FirstModel(96x96x1)-----------------------
learning_rate: 0.000300
training_epochs: 67
batch_size: 150
Cost: 0.000010
Accuracy: 0.685083 ↑
face 0 miss classification: 11, accuracy: 0.560000
face 1 miss classification: 6, accuracy: 0.933333
face 2 miss classification: 6, accuracy: 0.400000
face 3 miss classification: 11, accuracy: 0.153846
face 4 miss classification: 11, accuracy: 0.607143
face 5 miss classification: 3, accuracy: 0.250000
face 6 miss classification: 9, accuracy: 0.181818
Comment: "현 모델의 가장 높은 정확도를 뽑아낼 수 있는 learning rate는 0.0003, training epoch은 67.."

learning rate 0.0003, training epoch 72 -> accuracy 67.9%
learning rate 0.0003, training epoch 63 -> accuracy 66.2%
learning rate 0.0003, training epoch 67 -> accuracy 68.5% ★
learning rate 0.0001, training epoch 67 -> accuracy 65.1%
learning rate 0.0005, training epoch 67 -> accuracy 66.8%

