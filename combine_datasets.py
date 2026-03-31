import pandas as pd

# Load datasets
original_df = pd.read_csv("train.csv")
filtered_df = pd.read_csv("filtered_augmented_train.csv")

print("Original:", original_df.shape)
print("Filtered:", filtered_df.shape)

# Combine
final_df = pd.concat([original_df, filtered_df], ignore_index=True)

# Shuffle (important)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Final dataset size:", final_df.shape)

# Save
final_df.to_csv("final_train.csv", index=False)

print("Saved final_train.csv")