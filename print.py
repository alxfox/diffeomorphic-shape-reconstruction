    

import pickle

f = open('loss.pckl', 'rb')
loss= pickle.load(f)
loss1= pickle.load(f)
loss3= pickle.load(f)
print(loss)
print(loss1)
print(loss3)
f.close()