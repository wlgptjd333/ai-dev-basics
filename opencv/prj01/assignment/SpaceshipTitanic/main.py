import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 불러오기 및 합치기 (전처리 통일)
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
all_data = pd.concat([train, test], ignore_index=True)

# 2. 특징
# (1) 총 지출 금액 (Total Spend)
expenses = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in expenses:
    all_data[col] = all_data[col].fillna(0)
all_data['TotalSpend'] = all_data[expenses].sum(axis=1)

# (2) 객실 번호(Cabin) 분해 -> 층(Deck)과 좌우방향(Side)이 생존율에 엄청난 영향을 줌
all_data[['Deck', 'Num', 'Side']] = all_data['Cabin'].str.split('/', expand=True)

# 3. 결측치 채우기 및 숫자 변환
# 돈을 1원도 안 쓴 사람은 냉동 수면(CryoSleep) 중이었을 확률이 높다는 논리를 적용
all_data['CryoSleep'] = all_data['CryoSleep'].fillna(all_data['TotalSpend'] == 0).astype(int)
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
all_data['VIP'] = all_data['VIP'].fillna(False).astype(int)

# 4. 범주형(글자) 데이터를 숫자로 변환 (라벨 인코딩)
features_to_encode = ['HomePlanet', 'Destination', 'Deck', 'Side']
for col in features_to_encode:
    all_data[col] = pd.factorize(all_data[col])[0]

# 5. 데이터 다시 나누기
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'TotalSpend', 'Deck', 'Side'] + expenses
X_train = all_data[all_data['Transported'].notnull()][features]
y_train = all_data[all_data['Transported'].notnull()]['Transported'].astype(int)
X_test = all_data[all_data['Transported'].isnull()][features]

# 6. 학습
m = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
m.fit(X_train, y_train)

# 7. 예측 및 저장
result = m.predict(X_test)
test["Transported"] = result.astype(bool)
test[["PassengerId", "Transported"]].to_csv('data/submission.csv', index=False)