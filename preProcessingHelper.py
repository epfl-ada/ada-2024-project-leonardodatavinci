import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
from locationHelper import LocationHelper

class PreProcessRatings():
    """
    PreProcessing Class for the Rating dataframe. Use the method "get_dataframe" to get the dataframe with columns of your choice 


    Parameters
    ----------
    platform : one of ["BeerAdvocate", "RateBeer"]
    """
    
    RATE_BEER_URL = "./data/RateBeer/"
    BEER_ADVOCATE_URL = "./data/BeerAdvocate/"
    URL = None
    

    additional_cols = []
    all_cols = []
    dataset = None
    platform = None
    raw_df = None

    dtypes = {
    "beer_name": "str",
    "beer_id": "int",
    "brewery_name": "str",
    "brewery_id": "int",
    "style": "str",
    "abv": "float",
    "date": "int",
    "user_name": "str",
    "user_id": "str",
    "appearance": "float",
    "aroma": "float",
    "palate": "float",
    "taste": "float",
    "overall": "float",
    "rating": "float",
    "review": "bool"}

    def __init__(self, platform = "BeerAdvocate"):
        self.platform = platform
        
        # initial read of the raw reviews data
        if self.platform == "BeerAdvocate":
            self.URL = self.BEER_ADVOCATE_URL
        elif self.platform == "RateBeer":
            self.URL = self.RATE_BEER_URL

        # read the raw data
        df = self.read_df_init()
        self.raw_df = self.specify_dtypes(df)
        print("now you can get dataframes with the \"get_dataframe\" handle")

    
    def get_dataframe(self, dataset = "reviews", additinal_cols = []) -> pd.DataFrame:
        """
        Get pandas dataframe depending on the parameters 

        Parameters
        ----------
        dataset : one of ["ratings", "reviews"], only for BeerAdvocate as there are only reviews for RateBeer (reviews.txt and ratings.txt are the same)
            reviews are a subset of the ratings. ratings contain specific metrics where as the reviews only contain the column "rating"
            the default columns for "reviews" are
                "appearance",
                "aroma",
                "palate",
                "taste",
                "overall",
                "rating",

            and for "ratings":
                "rating"

        additional_cols : list with elements of 
            ["country_name",
            "country_code3" (country code with 3 letters),
            "country_code2" (country code with 3 letters),
            "continent",
            "state",
            "pycountry_object" (pycountry object),
            "date_object" (datetime object),
            "month",
            "beer_name",
            "beer_id",
            "brewery_id",
            "style", 
            "abv", 
            "date" (timestamp, int),
            "user_name",
            "user_id"]  
            where the location related objects depend on the users-profile location
        """
        self.dataset = dataset
        self.additional_cols = additinal_cols

        # load the current raw data of the object
        df = self.raw_df.copy()
        if self.platform == "BeerAdvocate":
            # only reviews or all the columns depending on the dataset choice
            df = self.reviews_or_ratings(df)
        else:
            # every column in RateBeer is a review
            self.dataset = "reviews"

        df = self.fill_datetime(df)
        df = self.fill_location(df)

        # depending on the dataset chosen we have different important columns
        all_cols = []
        if dataset == "reviews":
            all_cols = ["rating", "appearance", "aroma", "palate", "taste", "overall"] + self.additional_cols
        elif dataset == "ratings":
            all_cols = ["rating"] + self.additional_cols
        
        # return the final dataframe
        return df[all_cols]

    def read_df_init(self):

        # drop text?
        drop_text = not "text" in self.additional_cols

        with open(self.URL + 'ratings.txt', 'r') as file:
            print(f"start parsing the beer reviews for {self.platform}")
            content = file.read()

            # Split the content by blank lines into separate reviews
            review_blocks = content.strip().split('\n\n')

            # Parse each review and store in a list of dictionaries
            
            parsed_reviews = [self.parse_beer_rating(review, drop_text=drop_text) for review in review_blocks]
            print(f"finished parsing the beer reviews for {self.platform}")
        return pd.DataFrame(parsed_reviews)


      
    def parse_beer_rating(self, rating, drop_text = True):
        rating_data = {}
        for line in rating.split('\n'):
            key, value = line.split(':', 1)

            if drop_text and key == "text":
                continue

            rating_data[key.strip()] = value.strip()
        return rating_data
    
    
    def specify_dtypes(self, df: pd.DataFrame):

        print("start converting datatypes")
        if self.platform == "BeerAdvocate":
            # map review column from string to boolean
            df["review"] = df["review"].map(self.booleanConverter)
        elif self.platform == "RateBeer":
            # there is no review column for RateBeer
            self.dtypes.pop("review")
        print("end converting datatypes")
        return df.astype(self.dtypes)
    

    def reviews_or_ratings(self, df: pd.DataFrame):
        # decides between reviews or ratings based on the "dataset" value
        if self.dataset == "reviews":
            # only return cols for which review is true
            return df[df["review"]]
        elif self.dataset == "ratings":
            # drop all the values that are null for all the reviews
            return df.drop(["appearance", "aroma", "palate", "taste", "overall"], axis=1)
        else:
            raise ValueError(f"{self.dataset} is not a valid dataset")
            
        

    def fill_datetime(self, df: pd.DataFrame):
        # dataframe with date objects
        self.int_to_datetime(df)
        if "month" in self.additional_cols:
            self.datetime_to_month(df)
        if "year" in self.additional_cols:
            self.datetime_to_year(df)
        return df
    
    def fill_location(self, df: pd.DataFrame):
        loc_related_cols = ["country_name", "country_code3", "country_code2", "continent", "state", "pycountry_object"]
        if len(set(loc_related_cols) - set(self.additional_cols)) == len(loc_related_cols):
            # there are no location related cols
            return df
        
        users_df = pd.read_csv(self.URL + "users.csv")
        reviews_users_merged = pd.merge(df, users_df, how="left", on="user_id")

        lh = LocationHelper(reviews_users_merged["location"])
        
        if "country_name" in self.additional_cols:
            reviews_users_merged["country_name"] = lh.get_country_names()
        if "country_code3" in self.additional_cols:
            reviews_users_merged["country_code3"] = lh.get_country_codes3()
        if "country_code2" in self.additional_cols:
            reviews_users_merged["country_code2"] = lh.get_country_codes2()
        if "continent" in self.additional_cols:
            reviews_users_merged["continent"] = lh.get_continent_names()
        if "state" in self.additional_cols:
            reviews_users_merged["state"] = lh.get_state_names()

        #stats 
        nr_reviews = reviews_users_merged.shape[0]
        nr_locations = reviews_users_merged["location"].notna().sum()
        nr_locations_nan = reviews_users_merged["location"].isna().sum()
        print(f'From {nr_reviews} reviews, {nr_locations} have a location (corresponding to the user) and {nr_locations_nan} do not have a location')

        return reviews_users_merged
    


### HELPER METHODS ###

    def booleanConverter(self, x): 
        x = x.lower()
        if x == "false":
            return False
        elif x == "true":
            return True
        else:
            return np.nan
        
    def int_to_datetime(self, df: pd.DataFrame):
        df["date_object"] = df["date"].apply(datetime.fromtimestamp)

    def datetime_to_month(self, df: pd.DataFrame):
        try:
            df["month"] = df["date_object"].apply(lambda x: x.month)
        except KeyError:
            print("first create a datetime object column!")

    def datetime_to_year(self, df: pd.DataFrame):
        try:
            df["year"] = df["date_object"].apply(lambda x: x.year)
        except KeyError:
            print("first create a datetime object column!")