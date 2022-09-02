from regression import regression_func
import pandas as pd
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1204593914&single=true&output=csv')

predictions = regression_func(df)