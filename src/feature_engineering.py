"""
Feature Engineering Module

Creates features with strict leakage protection:
- All aggregations computed only on training data
- Time-aware: no future information used
- Handles unseen categories gracefully
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineer with storage for training statistics."""
        self.lane_stats = None
        self.pickup_stats = None
        self.delivery_stats = None
        self.equipment_stats = None
        self.global_stats = None
        self.start_date = None
        
    def fit(self, df, target_col='posted_rate'):
        """
        Compute aggregations from training data only.
        
        Parameters:
        -----------
        df : DataFrame
            Training data
        target_col : str
            Target column name
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        self.start_date = df['date'].min()
        
        # Global statistics
        self.global_stats = {
            'mean_rate': df[target_col].mean(),
            'median_rate': df[target_col].median(),
            'mean_rpm': (df[target_col] / df['distance']).mean()
        }
        
        # Lane-level statistics
        df['lane'] = df['pickup'] + ' -> ' + df['delivery']
        lane_agg = df.groupby('lane').agg({
            target_col: ['mean', 'std', 'count'],
            'distance': 'mean'
        })
        lane_agg.columns = ['lane_mean_rate', 'lane_std_rate', 'lane_count', 'lane_mean_distance']
        self.lane_stats = lane_agg
        
        # Pickup city statistics
        pickup_agg = df.groupby('pickup').agg({
            target_col: ['mean', 'std', 'count']
        })
        pickup_agg.columns = ['pickup_mean_rate', 'pickup_std_rate', 'pickup_count']
        self.pickup_stats = pickup_agg
        
        # Delivery city statistics
        delivery_agg = df.groupby('delivery').agg({
            target_col: ['mean', 'std', 'count']
        })
        delivery_agg.columns = ['delivery_mean_rate', 'delivery_std_rate', 'delivery_count']
        self.delivery_stats = delivery_agg
        
        # Equipment statistics by distance bucket
        df['distance_bucket'] = pd.cut(df['distance'], 
                                        bins=[0, 500, 1000, 1500, 5000],
                                        labels=['short', 'medium', 'long', 'very_long'])
        
        equip_dist_agg = df.groupby(['equipment', 'distance_bucket']).agg({
            target_col: 'mean'
        }).reset_index()
        equip_dist_agg.columns = ['equipment', 'distance_bucket', 'equip_dist_mean_rate']
        self.equipment_stats = equip_dist_agg
        
        return self
    
    def transform(self, df):
        """
        Apply feature engineering to new data.
        
        Parameters:
        -----------
        df : DataFrame
            Data to transform
            
        Returns:
        --------
        DataFrame : Data with engineered features
        """
        df = df.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # === CALENDAR FEATURES ===
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        # US Holidays 2025 (major freight impact days)
        holidays_2025 = [
            '2025-01-01',  # New Year's Day
            '2025-05-26',  # Memorial Day
            '2025-07-04',  # Independence Day
            '2025-09-01',  # Labor Day
            '2025-11-27',  # Thanksgiving
            '2025-12-25',  # Christmas
        ]
        holidays = pd.to_datetime(holidays_2025)
        df['is_holiday'] = df['date'].isin(holidays).astype(int)
        
        # Near holiday (3 days before/after)
        df['near_holiday'] = 0
        for holiday in holidays:
            df.loc[(df['date'] >= holiday - timedelta(days=3)) & 
                   (df['date'] <= holiday + timedelta(days=3)), 'near_holiday'] = 1
        
        # Thanksgiving week effect (huge freight demand)
        thanksgiving = pd.Timestamp('2025-11-27')
        df['thanksgiving_week'] = ((df['date'] >= thanksgiving - timedelta(days=7)) & 
                                   (df['date'] <= thanksgiving)).astype(int)
        
        # Christmas season (Dec 15-25)
        df['christmas_season'] = ((df['date'].dt.month == 12) & 
                                  (df['date'].dt.day >= 15)).astype(int)
        
        # Time trend (days since start)
        if self.start_date is None:
            raise RuntimeError('FeatureEngineer must be fitted before transform')
        df['days_since_start'] = (df['date'] - self.start_date).dt.days
        
        # === DISTANCE FEATURES ===
        df['log_distance'] = np.log1p(df['distance'])
        df['distance_squared'] = df['distance'] ** 2
        
        # Distance buckets
        df['distance_bucket'] = pd.cut(df['distance'], 
                                        bins=[0, 500, 1000, 1500, 5000],
                                        labels=['short', 'medium', 'long', 'very_long'])
        
        # === WEIGHT FEATURES ===
        df['log_weight'] = np.log1p(df['weight'])
        df['weight_per_mile'] = df['weight'] / df['distance']
        
        # Weight buckets
        df['weight_bucket'] = pd.cut(df['weight'], 
                                      bins=[0, 25000, 35000, 50000],
                                      labels=['light', 'medium', 'heavy'])
        
        # === INTERACTION FEATURES ===
        df['distance_x_weight'] = df['distance'] * df['weight'] / 1000000  # Scaled
        
        # Equipment indicators
        df['is_reefer'] = (df['equipment'] == 'Reefer').astype(int)
        df['is_flatbed'] = (df['equipment'] == 'Flatbed').astype(int)
        df['is_dryvan'] = (df['equipment'] == 'Dry Van').astype(int)
        
        # === LANE FEATURES (with fallback for unseen) ===
        df['lane'] = df['pickup'] + ' -> ' + df['delivery']
        
        # Merge lane statistics
        if self.lane_stats is not None:
            df = df.merge(self.lane_stats, left_on='lane', right_index=True, how='left')
            
            # Fallback for unseen lanes: use pickup + delivery averages
            df = df.merge(self.pickup_stats[['pickup_mean_rate']], 
                         left_on='pickup', right_index=True, how='left')
            df = df.merge(self.delivery_stats[['delivery_mean_rate']], 
                         left_on='delivery', right_index=True, how='left')
            
            # Fill missing lane stats with average of pickup and delivery
            df['lane_mean_rate'] = df['lane_mean_rate'].fillna(
                (df['pickup_mean_rate'] + df['delivery_mean_rate']) / 2
            )
            
            # If still missing (completely unseen pickup or delivery), use global
            df['lane_mean_rate'] = df['lane_mean_rate'].fillna(self.global_stats['mean_rate'])
            df['lane_std_rate'] = df['lane_std_rate'].fillna(0)
            df['lane_count'] = df['lane_count'].fillna(0)
            
            # Drop intermediate columns
            df = df.drop(columns=['pickup_mean_rate', 'delivery_mean_rate'], errors='ignore')
        
        # === PICKUP/DELIVERY FEATURES ===
        if self.pickup_stats is not None:
            df = df.merge(self.pickup_stats, left_on='pickup', right_index=True, how='left')
            df['pickup_mean_rate'] = df['pickup_mean_rate'].fillna(self.global_stats['mean_rate'])
            df['pickup_std_rate'] = df['pickup_std_rate'].fillna(0)
            df['pickup_count'] = df['pickup_count'].fillna(0)
        
        if self.delivery_stats is not None:
            df = df.merge(self.delivery_stats, left_on='delivery', right_index=True, how='left')
            df['delivery_mean_rate'] = df['delivery_mean_rate'].fillna(self.global_stats['mean_rate'])
            df['delivery_std_rate'] = df['delivery_std_rate'].fillna(0)
            df['delivery_count'] = df['delivery_count'].fillna(0)
        
        # === EQUIPMENT X DISTANCE FEATURES ===
        if self.equipment_stats is not None:
            df = df.merge(self.equipment_stats, 
                         on=['equipment', 'distance_bucket'], 
                         how='left')
            df['equip_dist_mean_rate'] = df['equip_dist_mean_rate'].fillna(self.global_stats['mean_rate'])
        
        # === DERIVED RATE FEATURES (using distance) ===
        # Estimated rate per mile based on lane history
        df['lane_rpm'] = df['lane_mean_rate'] / df['distance']
        df['lane_rpm'] = df['lane_rpm'].fillna(self.global_stats['mean_rpm'])
        
        # === COORDINATE FEATURES (if available) ===
        if 'pickup_lat' in df.columns and 'delivery_lat' in df.columns:
            df['pickup_delivery_lat_diff'] = df['delivery_lat'] - df['pickup_lat']
            df['pickup_delivery_lon_diff'] = df['delivery_lon'] - df['pickup_lon']
        else:
            # If coordinates not available, create dummy columns
            df['pickup_delivery_lat_diff'] = 0
            df['pickup_delivery_lon_diff'] = 0
        
        # Add lat/lon columns if missing (fill with 0)
        for col in ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
            if col not in df.columns:
                df[col] = 0
        
        # === MARKET FEATURES (if available) ===
        if 'market_index' not in df.columns:
            df['market_index'] = self.global_stats.get('mean_market_index', 1.0)
        
        if 'quote_signal' not in df.columns:
            df['quote_signal'] = self.global_stats.get('mean_quote_signal', 2.0)
        
        return df
    
    def fit_transform(self, df, target_col='posted_rate'):
        """Fit and transform in one step."""
        return self.fit(df, target_col).transform(df)

def get_feature_names(df):
    """Get list of feature columns (exclude metadata and target)."""
    exclude = ['load_id', 'posted_rate', 'date', 'pickup', 'delivery', 'lane', 
               'distance_bucket', 'weight_bucket']
    features = [col for col in df.columns if col not in exclude]
    return features
