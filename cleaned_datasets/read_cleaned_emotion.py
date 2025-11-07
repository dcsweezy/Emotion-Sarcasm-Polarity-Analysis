import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_Emotion_Detection_Data.csv")

# View basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# View first few samples
print("\nSample rows:\n", df.head())

# View label distribution
print("\nLabel distribution:")
print(df['label'].value_counts(normalize=True))
