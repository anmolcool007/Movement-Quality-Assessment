from matplotlib import pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = ['Rehabilitation \n  session assessment', 'Skeletal Joint \n Kinematics Analysis', 'MSK Telerehabilitation']
students = [26,22,8]
ax.pie(students, labels = langs,autopct='%1.2f%%', textprops={'family' : 'Times New Roman','weight' : 'bold','fontsize': 18})
plt.savefig('image.jpg', dpi=100, bbox_inches = 'tight')
plt.show()