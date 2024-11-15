import pandas as pd


class JeanneHelper:
    
    def assign_period(self, df, group_by, season_months=None, specific_month=None):
        """
        Helper function to assign periods (e.g., Month, Season, Day_in_Month, Year) to the dataframe.

        Parameters:
        - df: DataFrame with beer reviews.
        - group_by: How to group the data ('Month', 'Season', 'Day_in_Month', 'Year').
        - season_months: List of month numbers defining a season (used if group_by='Season').
        - specific_month: Specific month number to filter days within that month (used if group_by='Day_in_Month').

        Returns:
        - DataFrame with a new 'Period' column.
        """
        
        if group_by == 'Month':
            df['Period'] = df['Month']
            
        elif group_by == 'Season':
            if season_months is None:
                raise ValueError("Please provide 'season_months' as a list of months (e.g., [1, 2, 3]).")
            df = df[df['Month'].isin(season_months)]
            df['Period'] = f"Season {'-'.join(map(str, season_months))}"
            
        elif group_by == 'Day_in_Month':
            if specific_month is None:
                raise ValueError("Please provide a 'specific_month' (e.g., 1 for January) when grouping by day within a month.")
            df = df[df['Month'] == specific_month]
            df['Period'] = df['Day']
            
        elif group_by == 'Year':
            df['Period'] = df['Year']
            
        else:
            raise ValueError("Invalid group_by option. Choose 'Month', 'Season', 'Day_in_Month', or 'Year'.")
        
        return df

    def top_k_beer_styles(self, df, k=10, group_by='Month', season_months=None, specific_month=None):
        """
        Returns the top k most popular beer styles based on a specified grouping criterion.

        Parameters:
        - df: DataFrame with beer reviews.
        - k: Number of top beer styles to display per group.
        - group_by: How to group the data ('Month', 'Season', 'Day_in_Month', 'Year').
        - season_months: List of month numbers defining a season (used if group_by='Season').
        - specific_month: Specific month number to filter days within that month (used if group_by='Day_in_Month').

        Returns:
        - DataFrame with top k beer styles for each grouping.
        """
        # Assign periods using the helper function
        df = self.assign_period(df, group_by, season_months, specific_month)

        # Group by Period and Style to get total reviews
        grouped = df.groupby(['Period', 'Style']).agg(
            rating_count=('Rating', 'count')
        ).reset_index()

        # For each period, find the top k beer styles by total reviews
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
        - group_by: The period by which to group the data ('Month', 'Season', 'Day_in_Month', 'Year').
        - season_months: List of month numbers defining a season (used if group_by='Season').
        - specific_month: Specific month number to filter days within that month (used if group_by='Day_in_Month').

        Returns:
        - DataFrame containing the top k beer styles per period, with percentage values for each style.
        """
        
        # Assign periods (Month, Season, Day_in_Month, Year) to the dataframe
        df = self.assign_period(df, group_by, season_months, specific_month)
        
        # Group the data by the desired period and style, and calculate the number of ratings per style
        grouped = df.groupby([group_by, 'Style']).size().reset_index(name='rating_count')
        
        # Calculate the total number of ratings for each period
        total_ratings_per_period = grouped.groupby(group_by)['rating_count'].sum().reset_index(name='total_ratings')
        
        # Merge the total ratings with the grouped data
        grouped = pd.merge(grouped, total_ratings_per_period, on=group_by)
        
        # Calculate the percentage of total ratings for each style within the period
        grouped['percentage'] = (grouped['rating_count'] / grouped['total_ratings']) * 100
        
        # Sort by rating count and get the top k styles per period
        top_k_styles = grouped.groupby(group_by).apply(lambda x: x.nlargest(k, 'rating_count')).reset_index(drop=True)
        
        return top_k_styles
