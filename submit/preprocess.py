def getAgeBucket(age, intervals = [0, 18, 25, 35, 45, 50, 55, 1000]):
  assert age > 0
  bucket_index = None
  for i, cut_off in enumerate(intervals):
    if cut_off > age:
      return i - 1
  assert bucket_index and bucket_index in range(6)
  return bucket_index

def readUser(numberOfIntervals = 7, maxAge = 60, maxOccupation = 21):
  ret = {}
  with open("../files/user.txt", "r") as f:
    for l in f :
      app = [0 for i in range(2 + numberOfIntervals + maxOccupation)]
      if l[-1] == '\n' or l[-1] == '\r':
        l = l[:-1]
      l = l.split(",")
      if l[1] == "M" : 
        app[0] = 1
      elif l[1] == "F" :
        app[1] = 1
      try :
        age = int(l[2])
        app[2 + bucket_index] = 1 
      except:
        age = -1
      
      try:
        occ = int(l[3])
        app[2 + numberOfInterVals + occ] = 1
      except:
        occ = -1

      ret[int(l[0])] = app
  return ret

def readMovie(numberOfIntervals = 5, minAge = 1919, maxAge = 2000):
  ret = {}
  interval = float(maxAge - minAge + 1) / numberOfIntervals
  gen = []
  with open("../files/movie.txt", "r") as f:
    for l in f :
      if l[-1] == '\n' or l[-1] == '\r':
        l = l[:-1]
      l = l.split(",")
      if l[2] != "N/A":
        temp = l[2].split('|')
        for item in temp:
          gen.append(item)    
  gen = list(set(gen))
  maxGen = len(gen)
  with open("../files/movie.txt", "r") as f:
    for l in f :
      app = [0 for i in range(numberOfIntervals + maxGen)]
      if l[-1] == '\n' or l[-1] == '\r':
        l = l[:-1]
      l = l.split(",")
      try: 
        yr = int(l[1])
        app[(yr - minAge) / interval] = 1
      except:
        yr = 0

      if l[2] != "N/A":
        genre = l[2].split('|')
        for item in genre:
          app[numberOfIntervals + gen.index(item)] = 1
      ret[int(l[0])] = app
  return ret

def GetAvgRatingMap():
  user_mp = {}
  movie_mp = {} # id -> (total_rating, total_num)

  all_rating = 0
  all_count = 0

  with open('../files/train.txt') as f:
    for line in f:
      _, uid, mid, rating = line.strip().split(',')

      all_count += 1
      all_rating += int(rating)

      if uid not in user_mp:
        user_mp[uid] = (0, 0)
      curr_total_rating, curr_total_num = user_mp[uid]
      user_mp[uid] = (curr_total_rating + int(rating), curr_total_num + 1)

      if mid not in movie_mp:
        movie_mp[mid] = (0, 0)
      curr_total_rating, curr_total_num = movie_mp[mid]
      movie_mp[mid] = (curr_total_rating + int(rating), curr_total_num + 1)

  ret_user = {}
  for uid, (total_rating, total_num) in user_mp.iteritems():
    ret_user[int(uid)] = float(total_rating) / total_num
  ret_movie = {}
  for mid, (total_rating, total_num) in movie_mp.iteritems():
    ret_movie[int(mid)] = float(total_rating) / total_num 

  return ret_user, ret_movie, float(all_rating) / all_count

def readTrain():
  ret = []
  with open("../files/train.txt", "r") as f:
    for l in f:
      app = [0 for i in range(3)]
      if l[-1] == '\n' or l[-1] == '\r':
        l = l[:-1]
      l = l.split(",")
      app[0] = int(l[1])
      app[1] = int(l[2])
      app[2] = int(l[3])
      ret.append(app)
  return ret

def readTest():
  ret = []
  with open("../files/test.txt", "r") as f:
    for l in f:
      app = [0 for i in range(3)]
      if l[-1] == '\n' or l[-1] == '\r':
        l = l[:-1]
      l = l.split(",")
      app[0] = int(l[0])
      app[1] = int(l[1])
      app[2] = int(l[2])
      ret.append(app)
  return ret