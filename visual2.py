import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# Step 1: Load the dataset
data = pd.read_csv('archive\\data.csv')
data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Height', y='BMI', hue='Gender', palette='Set1', s=100, alpha=0.7)
plt.title('Scatter Plot of Height vs BMI (by Gender)', fontsize=16)
plt.xlabel('Height (cm)', fontsize=14)
plt.ylabel('BMI', fontsize=14)
plt.legend(title='Gender')
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Weight', y='BMI', hue='Gender', palette='Set1', s=100, alpha=0.7)
plt.title('Scatter Plot of Weight vs BMI (by Gender)', fontsize=16)
plt.xlabel('Weight (kg)', fontsize=14)
plt.ylabel('BMI', fontsize=14)
plt.legend(title='Gender')
plt.show()

data['Gender_encoded'] = data['Gender'].apply(lambda x: 0 if x == 'Male' else 1)

correlation_matrix = data[['Height', 'Weight', 'BMI', 'Gender_encoded']].corr()
print("Correlation Matrix:")
print(correlation_matrix)


bmi_male = data[data['Gender'] == 'Male']['BMI']
bmi_female = data[data['Gender'] == 'Female']['BMI']
f_stat, p_value = stats.f_oneway(bmi_male, bmi_female)

print(f"\nANOVA for BMI by Gender: F-statistic = {f_stat}, p-value = {p_value}")

height_male = data[data['Gender'] == 'Male']['Height']
height_female = data[data['Gender'] == 'Female']['Height']
f_stat_height, p_value_height = stats.f_oneway(height_male, height_female)

print(f"\nANOVA for Height by Gender: F-statistic = {f_stat_height}, p-value = {p_value_height}")

weight_male = data[data['Gender'] == 'Male']['Weight']
weight_female = data[data['Gender'] == 'Female']['Weight']
f_stat_weight, p_value_weight = stats.f_oneway(weight_male, weight_female)

print(f"\nANOVA for Weight by Gender: F-statistic = {f_stat_weight}, p-value = {p_value_weight}")


plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Height', y='Weight', hue='Gender', palette='Set1', s=100, alpha=0.7)
plt.title('Scatter Plot of Height vs Weight (by Gender)', fontsize=16)
plt.xlabel('Height (cm)', fontsize=14)
plt.ylabel('Weight (kg)', fontsize=14)
plt.legend(title='Gender')
plt.show()
data['Gender_encoded'] = data['Gender'].apply(lambda x: 0 if x == 'Male' else 1)
sns.pairplot(data, vars=['Height', 'Weight', 'BMI', 'Gender_encoded'], hue='Gender', palette='Set1', markers=["o", "s"])

plt.suptitle("Pair Plot of Height, Weight, BMI, and Gender", y=1.02, fontsize=16)
plt.show()