import pandas as pd

orig = pd.read_csv("train.csv")
div = pd.read_csv("filtered_diverse_train.csv")

print("Original:", orig.shape)
print("Diverse:", div.shape)

final = pd.concat([orig, div], ignore_index=True)
final = final.sample(frac=1, random_state=42).reset_index(drop=True)

final.to_csv("final_train_v2.csv", index=False)

print("Saved final_train_v2.csv")
print("Final size:", final.shape)