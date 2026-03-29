import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("prepared_meqsum.csv")

print("=" * 80)
print("LOADED PREPARED DATASET")
print("=" * 80)
print("Shape:", df.shape)


train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


print("\n" + "=" * 80)
print("SPLIT COMPLETE")
print("=" * 80)

print(f"Train size      : {len(train_df)}")
print(f"Validation size : {len(val_df)}")
print(f"Test size       : {len(test_df)}")

print("\nTrain sample:")
print(train_df.head(3))

print("\nValidation sample:")
print(val_df.head(3))

print("\nTest sample:")
print(test_df.head(3))

# -------------------------
# Save split datasets
# -------------------------
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nSaved files:")
print("- train.csv")
print("- validation.csv")
print("- test.csv")

print("\nStep 3 complete.")