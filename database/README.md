Database Backup

This notebook demonstrates how we convert our processed dataset (full_data.csv) into a SQLite database for backup and efficient data management.

The dataset contains ~500,000 entries, each consisting of a text sample and its corresponding emotion label.

Using pandas and sqlite3, we load the CSV, create a SQLite database (dataset.sqlite), and store the full dataset in a structured table format.

The notebook includes sample queries (e.g., retrieving 10 entries labeled as anger) to illustrate how the database can be used for quick exploration and validation.

Finally, the connection is closed to free up memory and keep the workflow clean.

This backup ensures our data is preserved in a robust and queryable format, making it easier to manage, share, and integrate into future experiments.
