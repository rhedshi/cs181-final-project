import numpy as np
import matplotlib.pyplot as plt

N1 = 3
avg1 = (0.781589, 0.848295, 0.810336)
std1 = (0.024949, 0.016554, 0.011468)

N2 = 4
avg2 = (0.475399, 0.604219, 0.711795, 0.679615)
std2 = (0.037151, 0.13359, 0.187379, 0.19308)

N3 = 3
avg3 = (0.97076, 0.97076, 0.935399)
std3 = (0.003289, 0.003289, 0.05053)

ind = np.arange(N2)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind + width/2, avg2, width, color='green', yerr=std2)

# add some
ax.set_ylabel('Coefficient of Determination (R^2)')	
ax.set_title('Score Regression')
ax.set_xticks(ind+width)
# ax.set_xticklabels( ('One vs. Rest', 'One vs. One', 'Logistic Regression') )
ax.set_xticklabels( ('Class 0', 'Class 1', 'Class 2', 'Class 3') )

plt.savefig('score_regression.png')
plt.show()