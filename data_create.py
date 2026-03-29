import pandas as pd


file_path = "MeQSum-master/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx"   

if file_path.endswith(".xlsx"):
    df = pd.read_excel(file_path)
elif file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    raise ValueError("Unsupported file format. Use .xlsx or .csv")

print("=" * 80)
print("ORIGINAL DATASET SHAPE:", df.shape)
print("=" * 80)


df = df[["CHQ", "Summary"]].copy()

df.rename(columns={"CHQ": "question", "Summary": "summary"}, inplace=True)

df = df.reset_index(drop=True)

print("\n" + "=" * 80)
print("PREPARED DATASET SHAPE:", df.shape)
print("=" * 80)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


df.to_csv("prepared_meqsum.csv", index=False)

print("\nPrepared dataset saved as: prepared_meqsum.csv")
print("Step 2 complete.")