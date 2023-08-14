'''
start plotting
'''

a = (m-s).values
b = m.values
c = (m+s).values
mean = b.mean()


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime

# Font size variable
font = 20

# Plotting
plt.figure(figsize=(16, 12))
# plt.figure(figsize=(12, 8))

dates = m.index
x_axis = pd.to_datetime(dates, format='%Y%m%d')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))

# plot first group with shadow
plt.plot(x_axis, b, label='Mean of R Squared', color='blue')
plt.fill_between(x_axis, a, c, color='blue', alpha=0.1)
plt.axhline(mean, color='red', linestyle='-', label='Mean across all dates')
plt.text(x_axis[-1]+timedelta(days=5), mean + 0.01, f"{mean:.2f}", verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=font*1.2)

# Adjusting font sizes with the font variable
plt.xlabel("Date", fontsize=font*1.2)
plt.ylabel("Out of sample R squared", fontsize=font*1.2)
plt.xticks(fontsize=font*1.2)
plt.yticks(fontsize=font*1.2)
plt.legend(fontsize=font*1.2)

plt.grid(True)

# Save the figure with the generated filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"plot_{timestamp}.pdf"
plt.savefig(path00+filename, dpi=1200, bbox_inches='tight', format='pdf')
plt.show()



'''
end   plotting
'''
