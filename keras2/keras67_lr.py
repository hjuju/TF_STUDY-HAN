weight = 0.5 # 첫번째는 임의의 숫자 줌
input = 0.5
lr = 0.01 # lr을 이용해서 goal_prediction이 나오게 함
goal_prediction = 0.8
epochs = 200

for iteration in range(epochs) :
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2  

    print("Error : " + str(error) + "\tprediction : " + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) ** 2   

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) ** 2

    if(down_error < up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight + lr