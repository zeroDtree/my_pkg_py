import numpy as np
from scipy import stats

data1 = np.array([5.1, 4.9, 5.0, 5.2, 4.8])
data2 = np.array([5.5, 5.6, 5.4, 5.3, 5.7])

t_statistic, p_value = stats.ttest_ind(data1, data2)
print("t 统计量:", t_statistic)
print("p 值:", p_value)