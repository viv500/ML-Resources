
import matplotlib.pyplot as plt # most of the libraries we need are in pyplot
import numpy as np
from matplotlib import style # extra styling

X_data = np.random.random(50) * 100 # list of 100 random values
y_data = np.random.random(50) * 100

plt.scatter(X_data, y_data, c="red", marker="*", alpha=0.3) #alpha is the transparency


# =============================
# TYPES OF CHARTS
# =============================

# default plot is LINE
weight = [213, 3424, 124, 34, 1241, 436, 234, 634]
height = [116, 1130, 300, 134, 185, 190, 122, 151]

plt.plot(weight, height)

# BAR CHART: categorized counts
# HISTOGRAM: Frequency ranges

languages = ["python", "C++", "C#", "Java", "Go", "Node.js", "React.Js", "Scala"]

plt.bar(languages, height, color="r", align="edge", width=0.5, edgecolor="blue") # needs to be strings , not numbers


ages = np.random.normal(20, 1.5, 1000)
# bins is categories

plt.hist(ages, bins=[ages.min(), 18, 19, 20, 21, ages.max()])

# Cumulative Histograms, do cumulative=True


favourite_language = [1, 13, 24, 53, 14, 62, 73, 2]

plt.pie(favourite_language, labels=languages)

# EXPLODING CERTAIN VALUES IN THE PIE CHART (MAKE THEM STAND OUT)
explodes = [0, 0, 0, 0, 0.2, 0, 0, 0.5]

plt.pie(favourite_language, labels=languages, explode=explodes)


# to show percentages
plt.pie(favourite_language, labels=languages, 
        autopct='%.2f%%') # 2 decimal places


# distance of percentage from center pctdistance=
# rotate the pie chart: startangle=(degrees)


height = np.random.normal(172, 8, 300)
plt.boxplot(height)

# =============================
# PLOT CUSTOMIZATION
# =============================

years = [2015, 2017, 2019, 2020, 2023]
incomes = [55, 63, 66, 75, 90]

plt.plot(years, incomes)
plt.title("Income of Vivek (in CAD)", fontsize=25, fontname="FreeSerif") # CHART TITLE
plt.xlabel("Year")
plt.xlabel("Income in CAD")


# But now, the x axis sayes "50", we want it to say 50k and introdcue new levels
# Y TICKS!

income_ticks = np.arange(10, 100, 5)
plt.yticks(income_ticks, [f"${x}k USD" for x in income_ticks])


# =============================
# LEGENDS
# =============================

stock1 = np.random.randint(20, 100,  size=(10))
stock2 = np.random.randint(5, 68,  size=(10))
stock3 = np.random.randint(35, 291,  size=(10))

plt.plot(stock1, label="Apple") # no x value means default is index in list
plt.plot(stock2, label="Google")
plt.plot(stock3, label="Microsoft")

# by default, it doesnt put these labels on the chart

# to get a legend
plt.legend(loc="upper left") # position of legenr, lower right etc

print(stock1)




# LEGEND FOR PIE CHART!
plt.pie(favourite_language, labels=None) #overwrite labels, so we can use a legend
plt.legend(labels=languages, loc="upper right")

# =============================
# STYLING
# =============================

# from matplotlib import style
style.use("dark_background")



# ====================================
# MULTIPLE FIGURES IN DIFFERENT TABS
# ====================================


plt.figure(1)
plt.scatter(X_data, y_data)

plt.figure(2)
plt.pie(favourite_language, labels=languages)



# ====================================
# SUBPLOTS (Plots in the same tab)
# ====================================

x = np.arange(100)

fig, axs = plt.subplots(2, 2) # 4 different plots
# name of plot is fig

axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title("Sine Wave") # HAS TO BE SET_TITLE, NOT TITLE
axs[0, 0].set_xlabel("VALUE")

axs[0, 1].plot(x, np.cos(x))
axs[0, 1].set_title("Cosine Wave") # HAS TO BE SET_TITLE, NOT TITLE
axs[0, 1].set_xlabel("VALUE")

axs[1, 0].plot(x, np.log(x))
axs[1, 0].set_title("Log Wave") # HAS TO BE SET_TITLE, NOT TITLE
axs[1, 0].set_xlabel("VALUE")

axs[1, 1].plot(x, np.exp(x))
axs[1, 1].set_title("Exponential Root") # HAS TO BE SET_TITLE, NOT TITLE
axs[1, 1].set_xlabel("VALUE")

# SET A SUPER TITLE FOR THE 4 CHARTS
fig.suptitle("MATHEATICAL FUNCTIONS")


# ================================
# EXPORTING PLOTS
# ================================

plt.savefig("fourplots.png", transparent=True, dpi=1000) #saves plt into a png, dpi is resolution


# ================================
# 3D, can rotate
# ================================
ax = plt.axes(projection="3d")

x = np.random.random(100)
y = np.random.random(100)
z = np.random.random(100)

ax.scatter(x, y, z)

plt.show()