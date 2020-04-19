import pandas as pd  # import pandas to use the panel data operations
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np  # use for basic math and NaN functions
from datetime import datetime, date #use to manipulate dates
import ast #use to evaluate and change data types
import gender_guesser.detector as gender #use to check the gender of a person based on their name

pd.options.display.max_columns = 500  # use to show whole dataframe

rio_df = pd.read_csv('./Airbnb_listing_info/Rio_BA.csv')  # import dataframe
print(rio_df.shape)  # show scope of dataframe

total_columns=list(rio_df.columns)# create a list of columns in order to show which features to be used
columns_for_deletion = ['listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url',
                        'xl_picture_url', 'host_id', 'host_url', 'host_location', 'host_about', 'host_thumbnail_url',
                        'host_picture_url', 'host_neighbourhood', 'host_listings_count', 'street', 'neighbourhood',
                        'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'smart_location', 'country_code',
                        'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
                        'maximum_maximum_nights',
                        'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_last_scraped', 'license',
                        'jurisdiction_names',
                        'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
                        'calculated_host_listings_count_private_rooms',
                        'calculated_host_listings_count_shared_rooms', 'minimum_minimum_nights',
                        'jurisdiction_names', 'square_feet']  # eliminate columns due to extraneous information

columns_for_boolean = ['require_guest_profile_picture', 'require_guest_phone_verification', 'requires_license',
                       'instant_bookable', 'is_business_travel_ready', 'is_location_exact', 'host_has_profile_pic',
                       'host_identity_verified', 'host_is_superhost', 'has_availability']  # create columns into a numeric boolean

columns_for_dummy = ['experiences_offered', 'cancellation_policy', 'bed_type', 'room_type',
                     'property_type']  # dummy these columns to be feed into model

columns_for_sentiment = ['name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit',
                         'access', 'interaction',
                         'house_rules']  # columns will be used in a length and sentiment analysis

rio_df = rio_df[[col for col in total_columns if
                 col not in columns_for_deletion]]  # use a list comprehension in order toeleminate unnicessary files
print(dict(rio_df.isna().sum())) #show all na values

na_fill_columns_for_characters = ['name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes',
                                  'transit', 'access','interaction', 'house_rules'] #columns that will be filled with a None

na_fill_in_for_zero = ['reviews_per_month', 'review_scores_value', 'review_scores_location',
                       'review_scores_communication',
                       'review_scores_checkin', 'review_scores_cleanliness', 'review_scores_accuracy',
                       'review_scores_rating', 'weekly_price', 'monthly_price',
                       'last_review', 'first_review', 'price', 'security_deposit', 'cleaning_fee',
                       'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
                       'host_total_listings_count'] #columns that will be zeroed out

na_fill_for_mode = ['bathrooms', 'bedrooms', 'beds'] #columns that will be iterated through with the mode

cols_for_price_fix = ['price', 'security_deposit', 'cleaning_fee'] #columns that will be fixed to be a integer price

for col in na_fill_columns_for_characters: # for each column in the character column fill list
    rio_df[col].replace(np.nan, 'None', inplace=True) # if the value is na the replace with none

for col in na_fill_in_for_zero: # for each column in the zero fill list
    rio_df[col].replace(np.nan, 0, inplace=True) # if the value is na replace with a zero

for col in na_fill_for_mode: # for a column in the mode fill list
    rio_df[col].fillna(value=rio_df[col].mode().iloc[0], inplace=True) #replace evey na with the mode of that column

print(dict(rio_df.isna().sum())) #show remaining na values

def price_col_fix(row): #instantiate the function to fix pricing
    return ast.literal_eval(row[1:].replace(',', '')) #return the value that is trimmed of the dollar sign and comma will return as a float

rio_df['price'] = rio_df['price'].apply(price_col_fix) #change the pricing column and pply the price fix function

def price_impute(df, days): #instantiate the pricing column to take a df for the price columns and the days 
    impute_list = df.values.tolist() #change this dataframe to a iterable list
    for row in impute_list: #go through each row in the list
        if row[1] == 0: #if the row is equal to zero
            row[1] = row[0]*days #multiple the row by the daily price and the number of days
        else: #if a price is present
            row[1] = ast.literal_eval(row[1][1:].replace(',', '')) #change this by taking out the dollar sign and the comma to be properly change
    return impute_list # provide back the new pricing

rio_df[['price', 'weekly_price']] = price_impute(rio_df[['price', 'weekly_price']], 7) #applys the function to calculate a weekly price based on seven days

rio_df[['price', 'monthly_price']] = price_impute(rio_df[['price', 'monthly_price']], 30) #applys the function to calculate a monthly price based on thrity days

additional_price_fix_cols = ['security_deposit', 'cleaning_fee', 'extra_people'] #shows the columns that also need pricing fixing

def additional_price_fix(col): #instantiates the price fixing column
    new_col = [] #provides an empty list
    for row in col: #goes through each row
        if row != 0: #checks if the row is not equal to zero
            row = ast.literal_eval(row[1:].replace(',', '')) #removes the dollar sign and comma to be placed into a float type
        else: #if the row is zero
            row = 0 #lets the row remain zero
        new_col.append(row) #adds the row to the new column
    return new_col #returns the new price column

for col in additional_price_fix_cols: #for each column in the additional pricing column
    rio_df[col] = additional_price_fix(rio_df[col]) # apply the additional price column to fix the pricing for these columns

def date_time_conversion(col): # isntantiates a new function to change the dates
    new_col = [] # adds a new list for iteration
    for row in col: #goes through the columns by row
        if row != 0: #if column is not equal to zero
            row = datetime.strptime(row, "%Y-%m-%d") #change from string to datetime
        else: #if it is zero
            row = 0 #row will be a numeric zero
        new_col.append(row) #add new row to list
    return new_col #return full list


df_dates = pd.DataFrame({'first_review': date_time_conversion(rio_df['first_review']),'last_review': date_time_conversion(rio_df['last_review'])}) #create a new data frame based on the dates of the columns
df_dates['days_between_reviews'] = (df_dates['last_review']-df_dates['first_review']) #create a new column that shows time between reviews

new_days = [] #creates an empty list iteration
for row in df_dates['days_between_reviews']: #iterates through the days between reviews
    if type(row) == int: #if the row is equal to an iteger
        row = 0 #change the row to a zero
    else: #if it is a datetime object
        row = row.days #takes the days
    new_days.append(row) #add row to list
df_dates['days_between_reviews'] = new_days # change the row to the new days list

df_dates.drop(columns=['first_review', 'last_review'], inplace=True) #drop the review dates
rio_df = rio_df.merge(df_dates, left_index=True, right_index=True) #add the number of days to the new data frame
rio_df.drop(columns=['first_review', 'last_review'], inplace=True) #remove those dates from the data frame

sia = SentimentIntensityAnalyzer() #instantiate sentiment analyzer

all_name_scores = []# add a empty list for names
name_corpus = list(set(rio_df['name'])) #iterate through the names as a list
for name in name_corpus: #for the row in the names list
    scores = sia.polarity_scores(name) #get the sentiment scores for the row
    all_name_scores.append(scores) #add the row to the list
names_sentiment = pd.DataFrame(all_name_scores) #create a data frame from the list
names_sentiment['name'] = name_corpus #add the names to the data frame
names_sentiment['name_length'] = names_sentiment['name'].str.len() # get the length of the names
names_sentiment.rename(columns={'neg': 'names_negative_sentiment_score','neu': 'names_neutral_sentiment_score',
                                'pos': 'names_positive_sentiment_score','compound': 'names_compound_sentiment_score'}, inplace=True) #rename columns for better understanding

rio_df = rio_df.merge(names_sentiment, on='name')# merge name sentiments to dataframe

all_summary_scores = [] #add empty summary score
summary_corpus = list(set(rio_df['summary'])) #create a iterable based on summary
for summary in summary_corpus:  # iterate through the summaries
    scores = sia.polarity_scores(summary) #score the summary
    all_summary_scores.append(scores) # add scores
summary_sentiment = pd.DataFrame(all_summary_scores) #create a scores dataframe
summary_sentiment['summary'] = summary_corpus #add summary to dataframe
summary_sentiment['summary_length'] = summary_sentiment['summary'].str.len()
summary_sentiment.rename(columns={'neg': 'summary_negative_sentiment_score','neu': 'summary_neutral_sentiment_score',
                                  'pos': 'summary_positive_sentiment_score','compound': 'summary_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(summary_sentiment, on='summary')

space_corpus = list(set(rio_df['space']))
all_space_scores = []
for space in space_corpus:
    scores = sia.polarity_scores(space)
    all_space_scores.append(scores)
space_sentiment = pd.DataFrame(all_space_scores)
space_sentiment['space'] = space_corpus
space_sentiment['space_number_of_words'] = space_sentiment['space'].str.len()
space_sentiment.rename(columns={'neg': 'space_negative_sentiment_score', 'neu': 'space_neutral_sentiment_score',
                                'pos': 'space_positive_sentiment_score', 'compound': 'space_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(space_sentiment, on='space')

all_description_scores = []
description_corpus = list(set(rio_df['description']))
for description in description_corpus:
    scores = sia.polarity_scores(space)
    all_description_scores.append(scores)
description_sentiment = pd.DataFrame(all_description_scores)
description_sentiment['description'] = description_corpus
description_sentiment['description_length'] = description_sentiment['description'].str.len()
description_sentiment.rename(columns={'neg': 'description_negative_sentiment_score','neu': 'description_neutral_sentiment_score',
                                      'pos': 'description_positive_sentiment_score','compound': 'description_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(description_sentiment, on='description')

all_neighborhood_overview_scores = []
neighborhood_overview_corpus = list(set(rio_df['neighborhood_overview']))
for neighborhood_overview in neighborhood_overview_corpus:
    scores = sia.polarity_scores(neighborhood_overview)
    all_neighborhood_overview_scores.append(scores)
neighborhood_overview_sentiment = pd.DataFrame(
    all_neighborhood_overview_scores)
neighborhood_overview_sentiment['neighborhood_overview'] = neighborhood_overview_corpus
neighborhood_overview_sentiment['neighborhood_length'] = neighborhood_overview_sentiment['neighborhood_overview'].str.len()
neighborhood_overview_sentiment.rename(columns={'neg': 'neighborhood_overview_negative_sentiment_score', 'neu': 'neighborhood_overview_neutral_sentiment_score',
                                                'pos': 'neighborhood_overview_positive_sentiment_score', 'compound': 'neighborhood_overview_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(neighborhood_overview_sentiment, on='neighborhood_overview')

all_notes_scores = []
notes_corpus = list(set(rio_df['notes']))
for notes in notes_corpus:
    scores = sia.polarity_scores(notes)
    all_notes_scores.append(scores)
notes_sentiment = pd.DataFrame(all_notes_scores)
notes_sentiment['notes'] = notes_corpus
notes_sentiment['notes_length'] = notes_sentiment['notes'].str.len()
notes_sentiment.rename(columns={'neg': 'notes_overview_negative_sentiment_score', 'neu': 'notes_neutral_sentiment_score',
                                'pos': 'notes_positive_sentiment_score','compound': 'notes_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(notes_sentiment, on='notes')

all_transit_scores = []
transit_corpus = list(set(rio_df['transit']))
for transit in transit_corpus:
    scores = sia.polarity_scores(transit)
    all_transit_scores.append(scores)
transit_sentiment = pd.DataFrame(all_transit_scores)
transit_sentiment['transit'] = transit_corpus
transit_sentiment['transit_length'] = transit_sentiment['transit'].str.len()
transit_sentiment.rename(columns={'neg': 'transit_overview_negative_sentiment_score', 'neu': 'transit_neutral_sentiment_score',
                                  'pos': 'transit_positive_sentiment_score','compound': 'transit_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(transit_sentiment, on='transit')

all_access_scores = []
access_corpus = list(set(rio_df['access']))
for access in access_corpus:
    scores = sia.polarity_scores(access)
    all_access_scores.append(scores)
access_sentiment = pd.DataFrame(all_access_scores)
access_sentiment['access'] = access_corpus
access_sentiment['access_number_of_words'] = access_sentiment['access'].str.len()
access_sentiment.rename(columns={'neg': 'access_negative_sentiment_score','neu': 'access_neutral_sentiment_score',
                                 'pos': 'access_positive_sentiment_score','compound': 'access_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(access_sentiment, on='access')

all_interaction_scores = []
interaction_corpus = list(set(rio_df['interaction']))
for interaction in interaction_corpus:
    scores = sia.polarity_scores(interaction)
    all_interaction_scores.append(scores)
interaction_sentiment = pd.DataFrame(all_interaction_scores)
interaction_sentiment['interaction'] = interaction_corpus
interaction_sentiment['interaction_number_of_words'] = interaction_sentiment['interaction'].str.len()
interaction_sentiment.rename(columns={'neg': 'interaction_negative_sentiment_score', 'neu': 'interaction_neutral_sentiment_score',
                                      'pos': 'interaction_positive_sentiment_score','compound': 'interaction_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(interaction_sentiment, on='interaction')

all_house_rules_scores = []
house_rules_corpus = list(set(rio_df['house_rules']))
for house_rules in house_rules_corpus:
    scores = sia.polarity_scores(house_rules)
    all_house_rules_scores.append(scores)
house_rules_sentiment = pd.DataFrame(all_house_rules_scores)
house_rules_sentiment['house_rules'] = house_rules_corpus
house_rules_sentiment['house_rules_number_of_words'] = house_rules_sentiment['house_rules'].str.len()
house_rules_sentiment.rename(columns={'neg': 'house_rules_negative_sentiment_score', 'neu': 'house_rules_neutral_sentiment_score',
                                      'pos': 'house_rules_positive_sentiment_score','compound': 'house_rules_compound_sentiment_score'}, inplace=True)

rio_df = rio_df.merge(house_rules_sentiment, on='house_rules')

rio_df.drop(columns=['summary', 'description', 'experiences_offered', 'neighborhood_overview', 'notes','transit', 'access', 'interaction', 'house_rules'], inplace=True) #drop columns from data frame
print(rio_df.shape)

gender = gender.Detector()#instantiate a gender dectector

host_gender = []
host_names = list(set(rio_df['host_name']))
host_names.remove(np.nan)
for name in host_names:
    host = gender.get_gender(name)
    host_gender.append(host)
gender_df = pd.DataFrame(host_gender, columns=['gender_by_name'])

gender_df['host_name'] = host_names
rio_df = rio_df.merge(gender_df, on='host_name')

rio_df.drop(columns=['host_name'], inplace=True)

rio_df['amenities'] = rio_df['amenities'].apply(
    lambda i: i[1:-1].replace('"', '').split(','))
unique_amenities = []
for row in rio_df['amenities']:
    for i in row:
        unique_amenities.append(i)
unique_amenities = list(set(unique_amenities))

unique_amenities.remove('')
unique_amenities.remove('translation missing: en.hosting_amenity_49')
unique_amenities.remove('translation missing: en.hosting_amenity_50')

for i in unique_amenities:
    rio_df['has_'+i] = rio_df['amenities'].apply(lambda unique_amenities: 1 if i in unique_amenities else 0)

rio_df['host_verifications'] = rio_df['host_verifications'].apply(
    lambda i: i[1:-1].replace('\'', '').split(','))

unique_verifications = []
for row in rio_df['host_verifications']:
    for i in row:
        unique_verifications.append(i)
unique_verifications = list(set(unique_verifications))

unique_verifications.remove('')

for i in unique_verifications:
    rio_df['verified_by_'+i] = rio_df['host_verifications'].apply(lambda unique_verifications: 1 if i in unique_verifications else 0)

rio_df.drop(columns=['host_verifications', 'amenities','calendar_updated'], inplace=True)

rio_df['market'][rio_df['market'] == 'Rio de Janeiro'] = 'Rio De Janeiro'

df_dummy_cols = ['cancellation_policy', 'bed_type', 'property_type','room_type', 'neighbourhood_cleansed',
 'market', 'host_response_time']
dummy_df = pd.get_dummies(rio_df[df_dummy_cols])
dummy_df.drop(columns=['host_response_time_0'], inplace=True)

rio_df = rio_df.merge(dummy_df, left_index=True, right_index=True)

rio_df['market_percentage'] = rio_df['market'].map(dict(rio_df['market'].value_counts(normalize=True)))

rio_df['host_response_time_percentage'] = rio_df['host_response_time'].map( dict(rio_df['host_response_time'].value_counts(normalize=True)))

rio_df['neighbourhood_cleansed_percentage'] = rio_df['neighbourhood_cleansed'].map(dict(rio_df['neighbourhood_cleansed'].value_counts(normalize=True)))

rio_df['property_type_percentage'] = rio_df['property_type'].map(dict(rio_df['property_type'].value_counts(normalize=True)))

rio_df['room_type_percentage'] = rio_df['room_type'].map(dict(rio_df['room_type'].value_counts(normalize=True)))

rio_df['bed_type_percentage'] = rio_df['bed_type'].map(dict(rio_df['bed_type'].value_counts(normalize=True)))

rio_df['cancellation_policy_percentage'] = rio_df['cancellation_policy'].map(dict(rio_df['cancellation_policy'].value_counts(normalize=True)))

rio_df['gender_by_name_percentage'] = rio_df['gender_by_name'].map(dict(rio_df['gender_by_name'].value_counts(normalize=True)))

for col in columns_for_boolean:
    rio_df[col] = rio_df[col].apply(lambda i: 1 if i == 't' else 0)

rio_df.to_csv('rio_df_clean_for_visualization.csv')

rio_df.drop(columns=['name', 'space', 'host_since', 'host_response_time', 'host_response_rate',
                     'neighbourhood_cleansed', 'market', 'country', 'latitude', 'longitude',
                     'property_type', 'room_type', 'bed_type', 'cancellation_policy'], inplace=True)

rio_df.to_csv('rio_df_clean_for_modeling.csv')

print(rio_df.head())
