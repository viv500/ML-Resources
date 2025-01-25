import pandas as pd
import numpy as np

# data frames are the main data structures of PD's
# uses numpy under the hood

df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"], index=["X", "Y"])
# by default, columns are 0 , 1, 2

# see the first 5 rows
df.head()
df.head(1) # gives first row

df.tail() # bottom 5

df.columns
df.index.to_list() # i.e. rows

df.info() # data types -> int64

df.describe() #list of aggregate functions like mean, median std, percntiles etc of the data 

df.nunique() # number of uniqiue values in each column

df.shape #same as NumPy, dimensions



# =====================================
# LOADING IN DATAFRAMES FROM CSV FILES
# =====================================

coffee = pd.read_csv("./coffee.csv")
results = pd.read_parquet("./results.parquet")
# olympics = pd.read_excel("./olympics-data.xlsx")

coffee.head()

# olympics.head()

# can alsp specific a specific sheet within the excel file
# olympics = pd.read_excel("./olympics-data.xlsx", sheet_name="results")

# can also convert types
# olympics = olympics.to_csv


# =====================================
# Accessing data
# =====================================

coffee.sample(10, random_state=1) # 10 random values, random state is to make it deter 

# accesing rows or columns
# coffee.loc[[0, 1, 2], [0, 1]] #prints rows 0, 1, 2 and cloumns 0 and 1 of those rows
# wont work cuz labels are words not indices, you can do that for iloc



# also works for ranges
coffee.loc[:, ["Day", "Coffee Type"]] # all rows, only columns 0 and 2
# coffee.loc[["latte", "espresso"]]



#iloc works the same, except it works with indices instead of labels


# can also modify data
# coffee.loc[1, "Monday"] = "Tuesday"


# both these can be used to find a list of all values in a colun
coffee.Day
coffee["Day"]

coffee.sort_values(["Units Sold", "Coffee Type"], ascending=False) # or ascending=[0, 1] one is ascending, othet is descending
# sort by units sold, if any clash, sort by coffee type in descending order



# =====================================
# Filtering data
# =====================================

bios = pd.read_csv("./bios.csv")
bios.info()

# filter for height greater than something
(bios.loc[bios["height_cm"] > 200])


#multiple filters (different way to look up)
bios[(bios['height_cm'] > 215) & (bios['born_country']=="USA")]
bios[(bios['name'].str.contains("John", case=False))] # case insensitive
# can also use regex , "John|Jeff|James"
#you can do regex=False to turn it off


# isin isat, i.e. is in list
print(bios[bios['born_country'].isin(["USA", "GB"])])


# =====================================
# Querry Function
# =====================================
print(bios.query('born_country=="USA" and born_city == "Seattle"'))


# =====================================
# Adding and Removing Columns
# =====================================
coffee['price'] = 10.99
# creates a new column with everythign set to 10.99


#specific prices based on row data
# using np.where CONDITIONALS
coffee['new_price']= np.where(coffee['Coffee Type'] == 'Espresso', 3.99, 4.99)
# if else statement

coffee.drop(columns=['price']) # does not modify
coffee.drop(columns=['price'], inplace=True)
# OR
# coffee.drop(columns=['price'])

# to hard copy, can't just set
coffee_new = coffee.copy()


# Feature engineering
coffee["Revenue"] = coffee["Units Sold"] * coffee["new_price"]


# Renaming Columns
coffee = coffee.rename(columns={'new_price': 'price'})

bios_new = bios.copy()
bios_new['first_name'] = bios_new['name'].str.split(' ').str[0]

#lambda functions
bios_new['height_category'] = bios_new['height_cm'].apply(lambda x: 'Short' if x < 165 else('Average' if x < 185 else 'Tall'))



# =====================================
# Merging and Concatenating Data
# =====================================