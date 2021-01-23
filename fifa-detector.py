# FIFA detector

import pandas as pd
from matplotlib import pyplot as plt

# 데이터 불러오기
data = pd.read_excel('./2015_7_data.xlsx', usecols = 'A,R,AT')
data_fifa = pd.read_csv('players_15.csv')

# 데이터 전처리

## data(일반인)에서 남자 데이터만 처리
data = data.rename({'ⓞ_02_성별':'sex', '①_003_키':'height', '①_031_몸무게':'weight'}, axis='columns')
data_male = (data['sex']=='남')
data= data[data_male]

data['height'] = data['height']/10

### 리스트로 변환
height = data['height'].values.tolist()
weight = data['weight'].values.tolist()

## data_fifa(축구선수) 키, 몸무게 추출 후 리스트로 변환
height_fifa = data_fifa['height_cm'].values.tolist()
weight_fifa = data_fifa['weight_kg'].values.tolist()

# 산점도
plt.scatter(height, weight)
plt.scatter(height_fifa, weight_fifa)

# k 최근접 이웃 모델링
random_num = 612
neighbors_num = 49

import random

## 데이터 랜덤 추출
height_random = random.sample(height, random_num)
weight_random = random.sample(weight, random_num)

height_fifa_random = random.sample(height_fifa, random_num)
weight_fifa_random = random.sample(weight_fifa, random_num)

## 일반인, 축구선수 데이터 합치기
height_total = height_random+height_fifa_random
weight_total = weight_random+weight_fifa_random

## 키, 몸무게 데이터를 2차원 리스트로 생성
people_data = [[h, w] for h, w in zip(height_total, weight_total)]

## 정답 데이터 리스트 생성
people_target = [0]*random_num + [1]*random_num


from sklearn.neighbors import KNeighborsClassifier

## 객체 생성
kn = KNeighborsClassifier(n_neighbors=neighbors_num)

## 훈련
kn.fit(people_data, people_target)

## 예측
kn.predict([[180, 80]])

## 정확도
kn.score(people_data, people_target)