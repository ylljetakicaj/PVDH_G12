import pandas as pd

def merge_neighbourhoods(listings, neighbourhoods):
    if 'neighbourhood_cleansed' in listings.columns:
        return listings.merge(neighbourhoods, left_on='neighbourhood_cleansed', right_on='neighbourhood', how='left')
    return listings

def merge_reviews(listings, reviews):
    review_counts = reviews.groupby('listing_id')['id'].count().reset_index().rename(columns={'id':'number_of_reviews'})
    reviews['comment_length'] = reviews['comments'].apply(lambda x: len(str(x)))
    avg_review_len = reviews.groupby('listing_id')['comment_length'].mean().reset_index().rename(columns={'comment_length':'avg_review_length'})
    
    df = listings.merge(review_counts, left_on='id', right_on='listing_id', how='left')
    df = df.merge(avg_review_len, left_on='id', right_on='listing_id', how='left')
    df.drop(columns=['listing_id_x','listing_id_y'], inplace=True, errors='ignore')
    return df
