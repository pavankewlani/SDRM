import pandas as pd
import numpy as np
import random
from datetime import datetime

# Read the existing CSV file
csv_file = "student dropout (1).csv"
df = pd.read_csv(csv_file)

print(f"Original dataset shape: {df.shape}")

# Create 500 rows with invalid and missing values
new_rows = []
columns = df.columns.tolist()

# Define possible invalid/missing values
invalid_values = [None, '', 'N/A', 'unknown', 'invalid', 'error', 999, -1, -999]
yes_no_values = ['yes', 'no', 'maybe', 'unknown', '', None]
school_values = ['GP', 'MS', 'INVALID', '', None, 123, 'XYZ']
gender_values = ['F', 'M', 'X', '', None, 'unknown']
address_values = ['U', 'R', 'INVALID', '', None, 'XYZ']
family_size_values = ['LE3', 'GT3', 'LARGE', '', None, 'SMALL', -1, 999]
parental_status_values = ['T', 'A', 'UNKNOWN', '', None, 'X', 1]
education_values = [1, 2, 3, 4, '', None, 'INVALID', -1, 999]

for i in range(500):
    row = {}
    for col in columns:
        # Randomly decide if this cell should be invalid/missing (30-70% chance)
        if random.random() < 0.5:
            # Use one of the invalid values
            if col == 'School':
                row[col] = random.choice(school_values)
            elif col == 'Gender':
                row[col] = random.choice(gender_values)
            elif col == 'Age':
                # Invalid ages
                row[col] = random.choice([None, '', -5, 999, 'A', 'invalid'])
            elif col == 'Address':
                row[col] = random.choice(address_values)
            elif col == 'Family_Size':
                row[col] = random.choice(family_size_values)
            elif col == 'Parental_Status':
                row[col] = random.choice(parental_status_values)
            elif col in ['Mother_Education', 'Father_Education']:
                row[col] = random.choice(education_values)
            elif col in ['School_Support', 'Family_Support', 'Extra_Paid_Class', 
                         'Extra_Curricular_Activities', 'Attended_Nursery', 
                         'Wants_Higher_Education', 'Internet_Access', 'In_Relationship', 'Dropped_Out']:
                row[col] = random.choice(yes_no_values)
            elif col in ['Travel_Time', 'Study_Time', 'Number_of_Failures', 
                         'Family_Relationship', 'Free_Time', 'Going_Out',
                         'Weekend_Alcohol_Consumption', 'Weekday_Alcohol_Consumption', 
                         'Health_Status', 'Number_of_Absences', 'Grade_1', 'Grade_2', 'Final_Grade']:
                # Invalid numeric values
                row[col] = random.choice([None, '', -999, 99999, 'X', 'invalid'])
            else:
                row[col] = random.choice(invalid_values)
        else:
            # Use valid data from existing dataset (copy from random existing rows)
            valid_rows = df[df[col].notna()]
            if len(valid_rows) > 0:
                row[col] = valid_rows[col].iloc[random.randint(0, len(valid_rows)-1)]
            else:
                row[col] = None
    
    new_rows.append(row)

# Create DataFrame from new rows
new_df = pd.DataFrame(new_rows)

# Append to existing CSV
result_df = pd.concat([df, new_df], ignore_index=True)

# Save back to CSV
result_df.to_csv(csv_file, index=False)

print(f"New dataset shape: {result_df.shape}")
print(f"Added 500 rows with invalid/missing values")
print(f"\nSample of added invalid data:")
print(result_df.tail(10))
