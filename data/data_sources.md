# Data Sources

Please download the following datasets to the `data/raw/` directory before running the notebooks:

### 1. IPL Matches and Deliveries
Source: Kaggle (e.g., [IPL Complete Dataset (2008-2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020))
Files to extract:
- `ipl_matches.csv`
- `ipl_deliveries.csv`

### 2. T20 World Cup
Source: Cricsheet (CSV Downloads -> T20s)
Direct URL: https://cricsheet.org/downloads/t20s_csv2.zip
Instructions: Extract all international T20 matches, filter by World Cup tournament, and collapse into:
- `t20wc_matches.csv`
- `t20wc_deliveries.csv`

### 3. Big Bash League (BBL)
Source: Cricsheet (CSV Downloads -> BBL)
Direct URL: https://cricsheet.org/downloads/bbl_csv2.zip
Instructions: Extract and merge BBL data into:
- `bbl_matches.csv`
- `bbl_deliveries.csv`
