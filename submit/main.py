import copy
import preprocess
import linear_regression
import numpy as np

def GetFeature(user, movie):
  ret = []
  for uf in user:
    ret.append(float(uf))
  for mf in movie:
    ret.append(float(mf))
  return ret

def main():
  user_avg_rating, moive_avg_rating, all_avg = preprocess.GetAvgRatingMap()

  user = preprocess.readUser()
  movie = preprocess.readMovie()

  train_data = preprocess.readTrain()

  test_data = preprocess.readTest()

  X = []
  y = []
  for line in train_data:
    uid, mid, rating = line
    user_feature = copy.deepcopy(user[uid])
    if uid in user_avg_rating:
      user_feature.append(user_avg_rating[uid])
    else:
      user_feature.append(all_avg)

    movie_feature = copy.deepcopy(movie[mid])
    if mid in moive_avg_rating:
      movie_feature.append(moive_avg_rating[mid])
    else:
      movie_feature.append(all_avg)

    features = GetFeature(user_feature, movie_feature)
    X.append(features)
    y.append(float(rating))

  print 'Begin training model:'
  model = linear_regression.LinearRegression()
  model.fit(X, y)
  print 'End training model.'

  test_X = []
  test_y = []
  for line in test_data:
    tid, uid, mid = line
    user_feature = copy.deepcopy(user[uid])
    if uid in user_avg_rating:
      user_feature.append(user_avg_rating[uid])
    else:
      user_feature.append(all_avg)

    movie_feature = copy.deepcopy(movie[mid])
    if mid in moive_avg_rating:
      movie_feature.append(moive_avg_rating[mid])
    else:
      movie_feature.append(all_avg)

    features = GetFeature(user_feature, movie_feature)

    y = model.predict(np.array([features]))
    test_y.append( (tid, int(round(y[0]))) )

  with open("../out/submit.txt", "w") as f:
    f.write("Id,rating\n")
    for item in test_y:
      f.write(str(item[0]) + "," + str(item[1]) + "\n")

if __name__ == '__main__':
  main()