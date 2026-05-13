# This module provides functions to load and validate CSV data for visualization purposes.
import csv
#load_csv function takes the file path and a ist of requires=d coloumns as input retuens a tuple of two list one with valid rows and one with bad rows
def load_csv(path: str, required_columns: list[str]) -> tuple[list[dict], list[tuple]]:
    valid_rows = []
    bad_rows = []
    try:
        with open(path, "r") as file:
            reader = csv.DictReader(file)
            # Check if the required columns are present in the CSV header
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no header.")
            # Check for missing required columns
            missing = [col for col in required_columns if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            for row_number, row in enumerate(reader, start=2):
                # Clean and validate the row data (basic cleaning, more complex validation can be added later)
                try:
                  cleaned = {k: (v.strip() if v is not None else "") for k, v in row.items()}
                    # validation logic to be added in the future because we don't know what the data will look like yet
                    valid_rows.append(cleaned)
                except Exception as e:
                    bad_rows.append((row_number, str(e), row))
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except ValueError:
        raise
    return valid_rows, bad_rows