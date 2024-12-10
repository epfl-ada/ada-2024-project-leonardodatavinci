import pandas as pd
import numpy as np


class JeanneHelper:
    
    def assign_period(self, df, group_by, season_months=None, specific_month=None):
        """
        Helper function to assign periods (e.g., month, Season, day_in_month, year) to the dataframe.

        Parameters:
        - df: DataFrame with beer reviews.
        - group_by: How to group the data ('month', 'Season', 'day_in_month', 'year').
        - season_months: List of month numbers defining a season (used if group_by='Season').
        - specific_month: Specific month number to filter days within that month (used if group_by='day_in_month').

        Returns:
        - DataFrame with a new 'Period' column.
        """
        
        if group_by == 'month':
            df['Period'] = df['month']
            
        elif group_by == 'Season':
            if season_months is None:
                raise ValueError("Please provide 'season_months' as a list of months (e.g., [1, 2, 3]).")
            df = df[df['month'].isin(season_months)]
            df['Period'] = f"Season {'-'.join(map(str, season_months))}"
            
        elif group_by == 'day_in_month':
            if specific_month is None:
                raise ValueError("Please provide a 'specific_month' (e.g., 1 for January) when grouping by day within a month.")
            df = df[df['month'] == specific_month]
            df['Period'] = df['day']
            
        elif group_by == 'year':
            df['Period'] = df['year']
            
        else:
            raise ValueError("Invalid group_by option. Choose 'month', 'Season', 'day_in_month', or 'year'.")
        
        return df

    def top_k_beer_styles(self, df, k=10, group_by='month', season_months=None, specific_month=None):
        """
        Returns the top k most popular beer styles based on a specified grouping criteria.

        Parameters:
        - df: DataFrame with beer reviews.
        - k: Number of top beer styles to display per group.
        - group_by: How to group the data ('month', 'Season', 'day_in_month', 'year').
        - season_months: List of month numbers defining a season (used if group_by='Season').
        - specific_month: Specific month number to filter days within that month (used if group_by='day_in_month').

        Returns:
        - DataFrame with top k beer styles for each grouping.
        """
        temp_df = df.copy()

        # Assign periods using the previous function
        temp_df = self.assign_period(temp_df, group_by, season_months, specific_month)

        # Group by Period and style to get the number of ratings
        grouped = temp_df.groupby(['Period', 'style']).agg(
            rating_count=('rating', 'count')
        ).reset_index()

        # For each period, find the top k beer styles by total ratings
        top_styles_per_period = (
            grouped.groupby('Period')
            .apply(lambda x: x.nlargest(k, 'rating_count'))
            .reset_index(drop=True)
        )

        return top_styles_per_period

    def top_k_beer_styles_percentage(self, df, k, group_by, season_months=None, specific_month=None):
        """
        This function calculates the top k beer styles based on the number of ratings within a specified period 
        (month, season, day in month, or year) and returns the percentage of ratings each style has within that period.

        Parameters:
        - df: DataFrame containing the beer reviews.
        - k: The number of top styles to return.
        - group_by: The period by which to group the data ('month', 'Season', 'day_in_month', 'year').
        - season_months: List of month numbers defining a season (used if group_by='Season').
        - specific_month: Specific month number to filter days within that month (used if group_by='day_in_month').

        Returns:
        - DataFrame containing the top k beer styles per period, with percentage values for each style.
        """
        temp_df = df.copy()

        # Assign period with previous methos
        temp_df = self.assign_period(temp_df, group_by, season_months, specific_month)
        
        # Group by desired period and style, and calculate the number of ratings per style
        grouped = temp_df.groupby([group_by, 'style']).size().reset_index(name='rating_count')
        
        # Calculate the number of ratings per period
        total_ratings_per_period = grouped.groupby(group_by)['rating_count'].sum().reset_index(name='total_ratings')
        
        grouped = pd.merge(grouped, total_ratings_per_period, on=group_by)
        
        # Calculate the percentage of total ratings for each style within the period
        grouped['percentage'] = (grouped['rating_count'] / grouped['total_ratings']) * 100
        
        # Sort by rating count and get the top k styles per period
        top_k_styles = grouped.groupby(group_by).apply(lambda x: x.nlargest(k, 'rating_count')).reset_index(drop=True)
        
        return top_k_styles
    
    def calculate_seasonality_score_by_style(self, df, group_by='style', min_reviews=100):
        """Calculate the seasonality score for the given dataset. 
        Seasonality score is the difference of rating between mean summer rating (6-8, june, july, august) 
        and mean winter rating (12-2; december, january, february) per beer style.

        Parameters:
        - df: pandas DataFrame containing beer review data with columns ['year', 'month', 'style', 'rating', 'Location'].
        - group_by: Column name to group by when calculating seasonality score (e.g., 'style', 'State').
        - min_reviews: Minimum number of reviews required in summer and winter combined for a group to be included.

        Returns:
        - pandas DataFrame with seasonality scores for each group (e.g., style, state) and year.
        """
        temp_df = df.copy()
        summer_months = [6, 7, 8]
        winter_months = [12, 1, 2]

        # Assign seasons
        temp_df['Season'] = temp_df['month'].apply(
            lambda x: 'Summer' if x in summer_months else ('Winter' if x in winter_months else None)
        )
        season_df = temp_df.dropna(subset=['Season'])

        # Keep only groups with at least `min_reviews` in summer and winter combined
        seasonal_review_counts = season_df.groupby(['year', group_by]).size().reset_index(name='Season_Review_Count')
        filtered_groups = seasonal_review_counts[seasonal_review_counts['Season_Review_Count'] >= min_reviews]

        season_df = season_df.merge(filtered_groups[['year', group_by]], on=['year', group_by], how='inner')

        # Calculate average ratings by season
        seasonal_stats = season_df.groupby(['year', group_by, 'Season']).agg(
            avg_rating=('rating', 'mean'),
        ).unstack(fill_value=0)

        # Compute seasonality score
        seasonal_stats['Seasonality_Score'] = (
            seasonal_stats['avg_rating']['Summer'] - seasonal_stats['avg_rating']['Winter']
        )

        return seasonal_stats[['Seasonality_Score']].reset_index()
    


def get_rating_per_month(df: pd.DataFrame):
    months = np.sort(df["month"].unique())
    all_months_vals = []
    for m in months:
        month_list = [rating for rating in df[df["month"] == m]["rating"].values]
        all_months_vals.append(month_list)
    return all_months_vals