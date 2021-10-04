import pandas as pd
import matplotlib.pyplot as plt

copynum = pd.read_csv('avecopynum.csv')

plt.hist(copynum['avecopynum'], bins=20)
plt.show()