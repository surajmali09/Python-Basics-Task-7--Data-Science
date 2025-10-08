import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset("titanic")

print(data.describe())
print(data.isnull().sum())


sns.histplot(data['age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.show()


corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()