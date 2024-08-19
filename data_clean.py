import pandas as pd

# Define the path to the large CSV file
file_path = 'output_with_vertical_similarity_fulllfw.csv'
output_path = 'cleaned_output_full.csv'

# Define the chunk size (number of rows to process at a time)
chunksize = 10**6  # Adjust the chunk size based on your system's memory capacity

# Initialize a list to hold cleaned chunks
cleaned_chunks = []

# Define a set to keep track of seen pairs
seen_pairs = set()

# Process the file in chunks
print("start cleaning")
for chunk in pd.read_csv(file_path, chunksize=chunksize):
    # Sort the file names alphabetically within each comparison pair in the chunk
    chunk['sorted_pair'] = chunk.apply(lambda row: tuple(sorted([row['File Name'], row['Compared To']])), axis=1)
    
    # Remove duplicate pairs within this chunk
    chunk_cleaned = chunk.drop_duplicates(subset=['sorted_pair'])

    # Filter out pairs that have already been seen in previous chunks
    chunk_cleaned = chunk_cleaned[~chunk_cleaned['sorted_pair'].isin(seen_pairs)]

    # Update the set of seen pairs
    seen_pairs.update(chunk_cleaned['sorted_pair'])

    # Drop the helper column
    chunk_cleaned = chunk_cleaned.drop(columns=['sorted_pair'])

    # Append the cleaned chunk to the list
    cleaned_chunks.append(chunk_cleaned)


print("concat data")
# Concatenate all the cleaned chunks into a single DataFrame
final_cleaned_data = pd.concat(cleaned_chunks, ignore_index=True)

print("saving data")
# Save the cleaned data to a new CSV file
final_cleaned_data.to_csv(output_path, index=False)

# Output the path where the cleaned data is saved
print(f"Cleaned data saved to: {output_path}")