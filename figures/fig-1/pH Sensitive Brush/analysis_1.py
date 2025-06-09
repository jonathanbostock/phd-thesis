### Jonathan Bostock

import numpy as np
import pandas as pd
import jbplot

def main():
    raw_data = pd.read_csv("2025-01-09 pH Response 1.csv")

    raw_data["pH"] = raw_data.apply(
	lambda row: float(row["Sample Name"].split(" ")[2]),
	axis=1)
    raw_data["DNA Present"] = raw_data.apply(
	lambda row: row["Sample Name"].split(" ")[0] != "Ctrl",
	axis=1)
    raw_data["Repeat"] = raw_data.apply(
	lambda row: float(row["Repeat"]),
	axis=1)
    
    """
    processed_data = pd.DataFrame()
    processed_data[["ph", "batch", "delta_d"]] = [
	[ph,
	 batch,
	 raw_data.where[raw_data["pH"] == ph && raw_data["Repeat"] == repeat && raw_data["DNA Present"] == True] \ 	 - raw_data.where[raw_data["pH"] == ph && raw_data["Repeat"] == repeat && raw_data["DNA Present"] == False]]
	for ph in [6.0, 6.5, 7.0, 7.5, 8.0]
	for repeat in range(1,4)]
    """

    processed_data_list = []
    for repeat in range(1,4):
        for ph in [6.0, 6.5, 7.0, 7.5, 8.0]:
            processed_data_list.append([
		ph,
		repeat,
		raw_data.where[raw_data["pH"] == ph & raw_data["Repeat"] == repeat & raw_data["DNA Present"] == True] - raw_data.where[raw_data["pH"] == ph & raw_data["Repeat"] == repeat & raw_data["DNA Present"] == False]])

if __name__ == "__main__":
    main()