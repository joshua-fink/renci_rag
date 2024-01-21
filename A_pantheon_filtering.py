import pandas as pd

person_df = pd.read_csv('data/person_2020_update.csv')
print(person_df.columns)

alive_df = person_df[person_df['alive']==True]
american_alive_df = alive_df[alive_df['bplace_country'] == "United States"]
american_women_alive_df = american_alive_df[american_alive_df['gender'] == "F"]
young_american_women_alive_df = american_women_alive_df[american_women_alive_df['birthyear'] > 1990]
young_american_women_alive_not_porn_df = young_american_women_alive_df[young_american_women_alive_df['occupation'] != 'PORNOGRAPHIC ACTOR']

print(young_american_women_alive_not_porn_df.shape)

young_american_women_alive_not_porn_df.to_csv('data/young_us_women_alive.csv', index=False)