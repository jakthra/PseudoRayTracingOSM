import os
from experimentlogger import load_experiments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

exp_root_path = 'exps/'
experiments = load_experiments(exp_root_path)

plt.style.use('seaborn')

fig = plt.figure(figsize=(10,5))
ax = plt.subplot(1,2,1)
minloss = 9
for exp in experiments:
    ax.plot(exp.results['test_loss'], label=exp.id)
    _loss = np.min(exp.results['test_loss'])
    _epoch =  np.argmin(exp.results['test_loss'])
    if _loss < minloss:
        minloss = _loss
        minepoch = _epoch
        minid = exp.id

ax.plot([minepoch], [minloss], 'o')


ax = plt.subplot(1,2,2)
for exp in experiments:
    ax.plot(exp.results['train_loss'], label=exp.id)


plt.legend()
plt.show()

print("Minimum test loss: {}, id: {}".format(minloss, minid))
