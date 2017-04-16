from sklearn.linear_model import LogisticRegression
import copy
import preprocess
import numpy as np

def main():
  user = preprocess.readUser()
  movie = preprocess.readMovie()

  train_data = preprocess.readTrain()

  test_data = preprocess.readTest()

  X = []
  y = []
  for line in train_data:
    features = []
    features.extend(user[line[0]])
    features.extend(movie[line[1]])
    X.append(features)
    y.append(line[2])

  print 'Begin training model:'
  model = LogisticRegression()
  model.fit(X, y)
  print 'End training model.'

  test_X = []
  test_y = []
  for line in test_data:
    features = []
    features.extend(user[line[1]])
    features.extend(movie[line[2]])
    y = model.predict(np.array(features).reshape(1, -1))
    test_y.append( (line[0], y[0]) )

  with open("../out/output.txt", "w") as f:
    f.write("Id,rating\n")
    for item in test_y:
      f.write(str(item[0]) + "," + str(item[1]) + "\n")

if __name__ == '__main__':
  main()