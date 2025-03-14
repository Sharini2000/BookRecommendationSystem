import mysql.connector
import pandas as pd
import ast  
import os

# Hardcoded for testing (replace password with your actual one)
MYSQL_HOST = "database-1.clqcu4ueotm0.us-east-2.rds.amazonaws.com"
MYSQL_USER = "admin"
MYSQL_PASSWORD = "BRS_Database"  # Replace with your RDS password
MYSQL_DATABASE = "BookReviews"

# Database configuration
db_config = {
    "host": MYSQL_HOST,
    "user": MYSQL_USER,
    "password": MYSQL_PASSWORD,
    "database": MYSQL_DATABASE,
    "port": 3306  # Explicitly specify the MySQL port
}

try:
    # Test connection
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    print("Connected to RDS successfully!")

    # Load CSV
    csv_file = "C:/Users/shari/Documents/Book recommender system/Backend/cleaned_books.csv"
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    # Safe conversion for lists in authors and categories
    def safe_convert(value):
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, list):
                    return ", ".join(parsed_value)
            except (SyntaxError, ValueError):
                pass 
        return value

    df["authors"] = df["authors"].apply(safe_convert)
    df["categories"] = df["categories"].apply(safe_convert)
    df = df.where(pd.notna(df), None)

    # Insert data
    for _, row in df.iterrows():
        sql = """INSERT INTO books (ID, Title, average_score, merged_reviews, authors, publisher, categories, publishedYear, helpfulness_score)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        values = (
            row["ID"], row["Title"], row["average_score"], row["merged_reviews"],
            row["authors"], row["publisher"], row["categories"],
            row["publishedYear"], row["helpfulness_score"]
        )
        cursor.execute(sql, values)

    conn.commit()
    print("CSV data successfully inserted into MySQL!")

except mysql.connector.Error as err:
    print(f"Database Error: {err}")
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()


# import mysql.connector
# import pandas as pd
# import ast  
# import os

# DB_PW = os.getenv("MYSQL_PASSWORD")
# MYSQL_HOST = os.getenv("MYSQL_HOST")
# MYSQL_DATABASE= os.getenv("MYSQL_DATABASE")
# MYSQL_USER= os.getenv("MYSQL_USER")

# db_config = {
#     "host": MYSQL_HOST,
#     "user": MYSQL_USER,        
#     "password": DB_PW,        
#     "database": "BookReviews"
# }

# conn = mysql.connector.connect(**db_config)
# cursor = conn.cursor()


# csv_file = "C:/Users/shari/Documents/Book recommender system/Backend/cleaned_books.csv"
# df = pd.read_csv(csv_file)


# df.columns = df.columns.str.strip()


# def safe_convert(value):
#     if isinstance(value, str):
#         try:
#             parsed_value = ast.literal_eval(value)  
#             if isinstance(parsed_value, list):
#                 return ", ".join(parsed_value)
#         except (SyntaxError, ValueError):
#             pass 
#     return value

# df["authors"] = df["authors"].apply(safe_convert)
# df["categories"] = df["categories"].apply(safe_convert)


# df = df.where(pd.notna(df), None)


# for _, row in df.iterrows():
#     sql = """INSERT INTO books (ID, Title, average_score, merged_reviews, authors, publisher, categories, publishedYear, helpfulness_score)
#              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
#     values = (
#         row["ID"],  
#         row["Title"],
#         row["average_score"],
#         row["merged_reviews"],
#         row["authors"],  
#         row["publisher"],
#         row["categories"], 
#         row["publishedYear"],
#         row["helpfulness_score"]
#     )
#     cursor.execute(sql, values)  


# conn.commit()
# cursor.close()
# conn.close()

# print("CSV data successfully inserted into MySQL!")
