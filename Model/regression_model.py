import sys
sys.path.insert(0,'../Data')
from data_cleaner import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %% Let's create some toy data
# Turn interactive mode on
plt.ion()
n_observations = 100
# return a subplot axes at the given grid position on the "canvas"
fig, ax = plt.subplots(1, 1)
# x axis is from -3 to 3 with n_observations evenly spaced
xs = np.linspace(-3, 3, n_observations)

print xs

print type(xs)


input_string_array = np.array(usable_data)
input_float_array = input_string_array.astype(float)
sanity_check_array = np.array(cleaned_data)



y_array = input_float_array[:, -1]
x_array = input_float_array[:, :-1]

print "Cleaned_data"
print sanity_check_array
# print "OG data"
# print usable_data[:10]
print "Float Data"
print input_float_array[:10]

print "x Values"
print x_array
print len(x_array)
print len(x_array[0])
print "y Values"
print y_array[:10]

print y_array.size