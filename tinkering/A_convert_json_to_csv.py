import json
import pandas as pd

def convert_json_to_csv(input_file_path: str, output_file_path: str) -> None:
    df = pd.read_json(input_file_path)
    df.to_csv(output_file_path, index=False)

convert_json_to_csv("data/bbc_news_list_uk.json", "data/bbc_news_list_uk.csv")
