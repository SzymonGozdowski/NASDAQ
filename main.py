import matplotlib.pyplot as plt
import pandas

data = pandas.read_csv('15m_data.csv', sep='\t', parse_dates=['DateTime'], dayfirst=False)
# print(data['High'])

plt.figure(1)
plt.plot(range(len(data)), data['High'])
plt.show()
