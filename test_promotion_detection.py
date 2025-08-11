import unittest
import pandas as pd
from io import StringIO

# Assuming the function is in a file named promotion_detection_logic.py
# For testing, I'll define it here directly.
def detect_promotions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects promotions in sales data based on sharp increases in Total_units.
    """
    df['Proc_date'] = pd.to_datetime(df['Proc_date'])
    df = df.sort_values(by=['store_id', 'item', 'Proc_date'])
    df['units_diff'] = df.groupby(['store_id', 'item'])['Total_units'].diff().fillna(0)
    stats = df.groupby(['store_id', 'item'])['units_diff'].agg(['mean', 'std']).reset_index()
    df = pd.merge(df, stats, on=['store_id', 'item'], suffixes=('', '_stats'))
    # Replace std with 0 if it's NaN (for items with few sales)
    df['std_stats'] = df['std_stats'].fillna(0)
    df['threshold'] = df['mean_stats'] + (2 * df['std_stats'])
    df['is_promotion'] = (df['units_diff'] > 0) & (df['units_diff'] > df['threshold'])
    df = df.drop(columns=['units_diff', 'mean_stats', 'std_stats', 'threshold'])
    return df

class TestPromotionDetection(unittest.TestCase):

    def test_promotion_detection(self):
        """Test that a clear promotion is detected correctly."""
        csv_data = """store_id,item,Proc_date,Total_units
104,3913116850,3/30/2023,1
104,3913116850,4/4/2023,1
104,3913116850,4/16/2023,1
104,3913116850,4/25/2023,10
104,3913116850,5/4/2023,1
"""
        sales_df = pd.read_csv(StringIO(csv_data))
        result_df = detect_promotions(sales_df)
        
        # The 4th entry (index 3) should be a promotion
        self.assertTrue(result_df.loc[3, 'is_promotion'])
        # Check that other entries are not promotions
        self.assertFalse(result_df.loc[0, 'is_promotion'])
        self.assertFalse(result_df.loc[1, 'is_promotion'])
        self.assertFalse(result_df.loc[2, 'is_promotion'])
        self.assertFalse(result_df.loc[4, 'is_promotion'])

    def test_no_promotion(self):
        """Test that no promotion is detected in steady sales data."""
        csv_data = """store_id,item,Proc_date,Total_units
104,3913116850,3/30/2023,5
104,3913116850,4/4/2023,6
104,3913116850,4/16/2023,5
104,3913116850,4/25/2023,7
104,3913116850,5/4/2023,6
"""
        sales_df = pd.read_csv(StringIO(csv_data))
        result_df = detect_promotions(sales_df)
        self.assertFalse(result_df['is_promotion'].any())

    def test_multiple_items_and_stores(self):
        """Test with multiple items and stores to ensure grouping works."""
        csv_data = """store_id,item,Proc_date,Total_units
104,A,1/1/2023,10
104,A,1/2/2023,12
104,A,1/3/2023,100
104,B,1/1/2023,5
104,B,1/2/2023,6
105,A,1/1/2023,20
105,A,1/2/2023,25
105,A,1/3/2023,22
"""
        sales_df = pd.read_csv(StringIO(csv_data))
        result_df = detect_promotions(sales_df)

        # Promotion for store 104, item A on 1/3/2023
        self.assertTrue(result_df.loc[(result_df['store_id'] == 104) & (result_df['item'] == 'A') & (result_df['Proc_date'] == pd.to_datetime('1/3/2023'))]['is_promotion'].iloc[0])
        
        # No promotion for store 104, item B
        self.assertFalse(result_df.loc[result_df['store_id'] == 104]['is_promotion'].iloc[0])
        
        # No promotion for store 105, item A
        self.assertFalse(result_df.loc[result_df['store_id'] == 105]['is_promotion'].any())

    def test_edge_case_single_sale(self):
        """Test that a single sale for an item is not a promotion."""
        csv_data = """store_id,item,Proc_date,Total_units
104,A,1/1/2023,50
"""
        sales_df = pd.read_csv(StringIO(csv_data))
        result_df = detect_promotions(sales_df)
        self.assertFalse(result_df['is_promotion'].iloc[0])

    def test_edge_case_two_sales(self):
        """Test with only two sales for an item."""
        csv_data = """store_id,item,Proc_date,Total_units
104,A,1/1/2023,10
104,A,1/2/2023,50
"""
        sales_df = pd.read_csv(StringIO(csv_data))
        result_df = detect_promotions(sales_df)
        # The second sale is a large jump, should be a promotion
        self.assertTrue(result_df['is_promotion'].iloc[1])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
