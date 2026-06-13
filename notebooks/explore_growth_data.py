import pandas as pd

# Load both datasets
lettuce = pd.read_csv("data/lollo_rosa_lettuce_synthetic.csv")
strawberry = pd.read_csv("data/strawberry_synthetic.csv")

print("=== LETTUCE ===")
print(lettuce.shape)
print(lettuce.columns.tolist())
print(lettuce.head(3))
print(lettuce.describe())

print("\n=== STRAWBERRY ===")
print(strawberry.shape)
print(strawberry.columns.tolist())
print(strawberry.head(3))
print(strawberry.describe())