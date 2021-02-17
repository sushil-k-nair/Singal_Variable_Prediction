from sklearn import linear_model
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('D:/tutorials/python_tuto/tutorial/exel/population_per_year1.csv')

print("Close graph window to preview PREDICTION")

plt.scatter(df.Year, df.Population)
plt.title("Population as per Year")
plt.xlabel("Year")
plt.ylabel("Population")
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[["Year"]], df.Population)

ans = reg.predict([[2022]])

print("{:.8f}".format(float(ans)))

