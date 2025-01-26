import numpy as np

# numpy has C style arrays udner the hood, more efficient than python
# optimized for linear algebra

#==================================
# ARRAY ATTRIBUTES
#==================================

a = np.array([1, 2, 3, 4])

print(type(a)) # <class 'numpy.ndarray'>

# can index and slice like python regular
# they are mutable like python

a_mul = np.array([[1, 2, 3], [4, 5, 6]])
# can print [][] like python

# NEW: SHAPE
print(a_mul.shape) # (2, 3)

# if its [[[1, 2][2, 6]][8, 6]]

test = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(test.shape) # (2, 2, 2)




print(a_mul.ndim) # dimension: 2 dimension
print(a_mul.size) # nummber of elements
print(a_mul.dtype) # tyes of data in array (32 bit int) -> since it was created in C

# an array [1, 2, "hello"] is an array of string
# type -> <U11 means string of less than 11 characters

#==================================
# ARRAY DATA TYPES
#==================================

# you can typecast arrays
a = np.array([1, 2, 3], dtype=np.int32)
print(a) # works
# if one of them was a word string : error. if "5" -> implicit conversion

# if you do not have a primitive data type (etc a dictionary) as a list member, dtype of list is object 


#==================================
# FILLING ARRAYS
#==================================

a = np.full((2, 3, 4), 9)
print(a) # i.e. create an array of dimensions 2, 3, 4 with all 9's

np.zeros((10, 5, 2))
np.ones((10, 9, 8))

np.empty((5, 5, 5)) # uninitialized array, instead of 0s. 

x_values = np.arange(0, 1000, 5) # array of numbers between 0 and 1000 with step size 5, excludes upper bound
x_values = np.linspace(0, 1000, 2) # evenly spread 2 values between 0 and 999


#==================================
# NAN AND INF
#==================================

### INFINITY AND NAN
print(np.nan)
print(np.inf)

# checking if a dataset has empty values:
# print(np.isnan(np.sqrt(-1))) # will print true
# print(np.isinf(10/0)) # will print true

#==================================
# MATHEMATICAL OPERATIONS
#==================================

# COMPARE WITH REGULAR PYTHON Lists
l1 = [1, 2, 3]
l2 = [4, 5, 6]

a1 = np.array(l1)
a2 = np.array(l2)

print(l1 * 5) # repeats same elements 5 times
print(a1 * 5) # multiplies each element by 5 (vector)
#print(l1 + 5) # error
print(a1 * 5) # adds 5 to each element (vector)


# can multiply and divide lists
print(a1 * a2) # same dimension, elements multiplied


#==================================
# VECTOR ADDITION AND MULTIPLICATION
#==================================
x = np.array([1, 2, 3]) # 1 x 3
y = np.array([[1],[2]]) # 2 x 1
print(x * y) # 2 x 3

a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sqrt(a)) # square roots each element
print(np.cos(a))
print(np.sin(a))
print(np.exp(a))
print(np.log(a))
print(np.log2(a))
print(np.log10(a))


#==================================
# ARRAY METHODS
#==================================

a = np.array([1, 2, 3])
np.append(a, [7, 8, 9]) # will return the appended array BUT WONT ACTUALLY MAKE THE CHANGE !!!
a = np.append(a, [7, 8, 9]) # makes the change

a = np.insert(a, 3, [10, 11, 12]) # insert that array into the third position

print(a)
# deleting
# print(np.delete(a, 1)) # deletes index 1 element regardless of nesting
# print(np.delete(a, 1, 0)) # deletes 2nd ROW (index 1)
# print(np.delete(a, 1, 1)) # deletes 2nd COLUMN (index 1)

#==================================
# STRUCTURING METHODS
#==================================

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(a.shape)

# reshape: all parameters have to multiply to exactly total number of elements
print(a.reshape(4, 2))
print(a.reshape(8)) # 8 elements
print(a.reshape(8, 1)) # 8 lists with 1 element each (i.e. 20 rows)
print(a.reshape(2, 2, 2)) # 2 collections, with 2 arrays each, with 2 elements eac

# resize vs reshape
a.resize((4, 2)) # modifies a
a.reshape((4, 2)) # does not modify a

# flattening
print(a.flatten()) # returns a flattened copy: deep copy
print(a.ravel()) # returns a flattened view: shallow copy

# a.flat gets rid of nesting

a = np.array([[1, 2, 3], [4, 5, 6]])
var = [number for number in a.flat]
print(var)


# TRANSPOSING
print(a.transpose) #swapping the axes
print(a.T) # does the same exact thing
print(a.swapaxes(0, 1)) # same thing, but useful for HIGH-DIMENSION arrays where you only want to swap 0 and 1

#==================================
# CONCATENATION, STACKING, SPLITTING
#==================================

#joining arrays
a1 = np.array([[1, 2, 3], 
              [4, 5, 6]])
a2 = np.array([[7, 8, 9], 
              [10, 11, 12]])

a = np.concatenate((a1, a2), axis = 0) # adds rows [7, 8, 9] and [10, 11, 12] to the end
a = np.concatenate((a1, a2), axis = 1) # first row is now [1, 2, 3, 7, 8, 9]

# stacking 
a = np.stack((a1, a2)) # creates a new dimension, and encloses a1 and a2 as 2 elements[a1, a2]
a = np.vstack((a1, a2)) #axis 0
a = np.hstack((a1, a2)) # axis 1

# splitting
print(np.split(a, 2, axis=0)) # splitting the rows into 2 arrays in a big array. [[], [], [], []] -> [[[], []], [[], []]]

#==================================
# AGGREGATE FUNCTIONS
#==================================

print(a.min()) # smallest value
print(a.max())
print(a.mean())
print(a.std())
print(a.sum())
print(np.median(a))

#==================================
# NUMPY RANDOMS
#==================================

number = np.random.randint(100) # exclusive of 10
print(number)

# generating random arrays
numbers1 = np.random.randint(100, size=(2, 3, 4))
print(numbers1)


# can also specific range for randnt
numbers2 = np.random.randint(1, 2, size=(3, 3)) #includes 1
print(numbers2)

# can also do it on distributions like binomial
numbers = np.random.binomial(10, p=0.35, size=(5, 10))



# RANDOM CHOICES
numbers = np.random.choice([1, 2, 3, 4, 5]) #or
numbers = np.random.choice([1, 2, 3, 4, 5], size=(2, 2))
print(numbers)

#==================================
# EXPORTING AND IMPORTING
#==================================


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

a = np.save("my-array.npy", a) # encodes a in binary and saves it to a file

# now even if you comment out the array declaration and save, you can load it

a = np.load("my-array.npy")
print(a)


# WORKS FOR CSV TOO
np.savetxt("my-array.csv", a, delimiter=",")




# !!!
# [] is an np array
# [[]] is a pandas dataframe
# [[]].values is an np array
