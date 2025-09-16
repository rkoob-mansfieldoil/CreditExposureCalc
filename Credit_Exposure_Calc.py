# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:22:31 2025

@author: DuPonMa8510
"""

"""
VAR analysis script with optional "Inputs" database.

- Tries to load contract inputs from an Inputs DB (service "InputsDB" in keyring).
- If not available, falls back to the hard-coded defaults.
- Keeps existing EntradeMPGProd DB usage and caching logic unchanged.

Configure keyring for InputsDB like:
    keyring.set_password("InputsDB", "username", "<username>")
    keyring.set_password("InputsDB", "password", "<password>")
    keyring.set_password("InputsDB", "server", "yourinputsdb.database.windows.net")
    keyring.set_password("InputsDB", "database", "InputsDB")

Expected table example (one-row):
    dbo.ContractInputs(id, start_month_input, Billing_Type, state, terms, Monthly_Volume_12Months)
    - terms and Monthly_Volume_12Months are JSON arrays or comma-separated strings.

If the inputs DB is not present or the row is missing/malformed, the script will use the built-in defaults.
"""

import numpy as np
import pandas as pd
import keyring
from sqlalchemy import create_engine, text
from functools import lru_cache
import time
import datetime
import os
import json
from datetime import datetime
import logging # <<< ADDED: Import logging module

# --- Logging Configuration ---
# <<< ADDED: Set up basic logging to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ---

# --------------------------
# Defaults - will be overridden if inputs DB is configured & returns values
# --------------------------

# --------------------------
# Hub Configuration
# --------------------------
# Curve UIDs for different hubs
HUB_CURVE_IDS = {
    "hub": "30000",  # Henry Hub
    "socal": "31217",
    "chi": "31198",
    "z6": "31338"
}

class FlexibleStateDict(dict):
    def __getitem__(self, key):
        if key is None:
            return "hub"  # default
        
        # Try exact match first
        if key in self:
            return super().__getitem__(key)
        
        # Try case-insensitive match
        key_lower = str(key).strip().lower()
        for state_key, hub_value in self.items():
            if state_key.lower() == key_lower:
                return hub_value
        

# Wrap your existing dictionary
STATE_TO_HUB = FlexibleStateDict({
    # Northeast/Z6 states
    "West Virginia": "z6",
    "Virginia": "z6",
    "New Jersey": "z6",
    "South Carolina": "z6",
    "North Carolina": "z6",
    
    # Midwest/Chicago hub states
    "Illinois": "chi",
    "Indiana": "chi",
    "Michigan": "chi",
    "Ohio": "chi",
    "Iowa": "chi",
    "Missouri": "chi",
    
    # SoCal states
    "California": "socal",
    
    # Default to Henry Hub for other states
    "Texas": "hub",
    "Florida": "hub",
    "FLORIDA": "hub",
    "Georgia": "hub",
    "Oklahoma": "hub",
})

# --------------------------
# JSON Encoder for NumPy Arrays
# --------------------------
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and other special types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"_numpy_array": obj.tolist(), "shape": obj.shape}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return {"_datetime": obj.strftime("%Y-%m-%d %H:%M:%S")}
        elif isinstance(obj, pd.Timestamp):
            return {"_timestamp": obj.strftime("%Y-%m-%d %H:%M:%S")}
        return super().default(obj)

def numpy_json_decoder(dct):
    """Custom JSON decoder that restores numpy arrays and other special types"""
    if "_numpy_array" in dct:
        arr = np.array(dct["_numpy_array"])
        if "shape" in dct and len(dct["shape"]) > 0:
            arr = arr.reshape(dct["shape"])
        return arr
    elif "_datetime" in dct:
        return datetime.strptime(dct["_datetime"], "%Y-%m-%d %H:%M:%S")
    elif "_timestamp" in dct:
        return pd.Timestamp(dct["_timestamp"])
    return dct

# --------------------------
# Monthly Data Cache Manager with Text Files
# --------------------------
class MonthlyDataCache:
    def __init__(self, cache_dir="var_cache"):
        """Initialize the cache manager with specified directory"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_price_cache_path(self, hub_curve_id, use_basis):
        """Generate the cache file path for price data (CSV)"""
        return os.path.join(self.cache_dir, f"price_data_{hub_curve_id}_{use_basis}.csv")
    
    def _get_derived_cache_path(self, hub_curve_id, use_basis):
        """Generate the cache file path for derived calculations (JSON)"""
        return os.path.join(self.cache_dir, f"derived_data_{hub_curve_id}_{use_basis}.json")
    
    def _get_metadata_path(self, hub_curve_id, use_basis):
        """Generate the cache file path for metadata (JSON)"""
        return os.path.join(self.cache_dir, f"metadata_{hub_curve_id}_{use_basis}.json")
    
    def should_refresh(self, hub_curve_id, use_basis):
        """
        Check if cache needs to be refreshed based on:
        1. If cache file doesn't exist.
        2. If the script is run for the first time in a new month.
        """
        cache_path = self._get_price_cache_path(hub_curve_id, use_basis)
        
        # 1. If cache doesn't exist, refresh needed.
        if not os.path.exists(cache_path):
            return True
            
        # Get current date
        today = datetime.now().date()
        
        # Get cache modification time
        mtime = os.path.getmtime(cache_path)
        cache_date = datetime.fromtimestamp(mtime).date()
        
        # 2. Refresh if the cache is from a previous month or year.
        if cache_date.year < today.year or cache_date.month < today.month:
            return True
        
        return False
    
    def get_price_data(self, get_db_func, hub_curve_id, use_basis):
        """
        Get price data with caching (saved as CSV)
        
        Parameters:
        get_db_func: Function to call to get data from database if needed
        hub_curve_id: The curve ID for the hub
        use_basis: If True, add basis curve to HH; if False, use hub curve directly
        
        Returns:
        DataFrame: Price data
        """
        cache_path = self._get_price_cache_path(hub_curve_id, use_basis)
        metadata_path = self._get_metadata_path(hub_curve_id, use_basis)
        
        if self.should_refresh(hub_curve_id, use_basis):
            logging.info(f"Refreshing price data for hub {hub_curve_id} from database...") # <<< MODIFIED
            # Get fresh data from database
            df = get_db_func(hub_curve_id, use_basis)
            
            # Save to CSV cache
            df.to_csv(cache_path, index=False)
            
            # Save metadata
            metadata = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "hub_curve_id": hub_curve_id,
                "use_basis": use_basis,
                "column_types": {col: str(df[col].dtype) for col in df.columns}
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logging.info(f"Price data cached to {cache_path}") # <<< MODIFIED
            logging.info(f"Metadata saved to {metadata_path}") # <<< MODIFIED
            return df
        else:
            # Load from CSV cache
            try:
                logging.info(f"Loading cached price data for hub {hub_curve_id} from {cache_path}...") # <<< MODIFIED
                
                # Load metadata first to get column types
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Create dtype dictionary for proper type conversion
                dtypes = {}
                datetime_cols = []
                
                for col, dtype_str in metadata["column_types"].items():
                    if "datetime" in dtype_str.lower():
                        datetime_cols.append(col)
                    elif "float" in dtype_str.lower():
                        dtypes[col] = float
                    elif "int" in dtype_str.lower():
                        dtypes[col] = int
                
                # Read CSV with proper types
                df = pd.read_csv(cache_path, dtype=dtypes)
                
                # Convert datetime columns
                for col in datetime_cols:
                    df[col] = pd.to_datetime(df[col])
                    
                return df
                
            except Exception as e:
                logging.error(f"Error loading cache, refreshing from database: {e}") # <<< MODIFIED
                df = get_db_func(hub_curve_id, use_basis)
                
                # Save to CSV cache
                df.to_csv(cache_path, index=False)
                
                # Save metadata
                metadata = {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "hub_curve_id": hub_curve_id,
                    "use_basis": use_basis,
                    "column_types": {col: str(df[col].dtype) for col in df.columns}
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                return df
    
    def get_derived_calculations(self, price_data, build_matrices_func, hub_curve_id, use_basis, recalculate=False):
        """
        Get pre-calculated matrices and derived values from price data (saved as JSON)
        Will calculate and cache if not already cached or recalculate is True
        
        Parameters:
        price_data: DataFrame with price data
        build_matrices_func: Function to build price matrices
        hub_curve_id: The curve ID for the hub (used for caching)
        use_basis: If True, basis was used (used for caching)
        recalculate: Force recalculation even if cache exists
        
        Returns:
        dict: Derived calculations including matrices and correlation
        """
        # <<< MODIFIED START: This is the fix.
        # The original code was overwriting the hub_curve_id and use_basis parameters by
        # inferring them from the price_data dataframe. This was incorrect because the
        # price_data for basis hubs also contains the Henry Hub '30000' CurveUID,
        # causing the logic to always default to Henry Hub.
        # By removing the overwriting lines, we now respect the correct parameters
        # passed into this function from `calculate_var_metrics`.
        
        # hub_curve_id = str(price_data['CurveUID'].iloc[0]) if 'CurveUID' in price_data.columns else "30000" # <<< DELETED
        # use_basis = hub_curve_id != "30000" # <<< DELETED
        
        # <<< MODIFIED END
        
        cache_path = self._get_derived_cache_path(hub_curve_id, use_basis)
        
        if recalculate or not os.path.exists(cache_path) or self.should_refresh(hub_curve_id, use_basis):
            logging.info(f"Calculating derived matrices from price data for hub {hub_curve_id}...") # <<< MODIFIED
            start_time = time.time()
            
            # Build matrices 
            HUB_Ln_Changes, dates, seqs = build_matrices_func(price_data)
            HUB_Ln_Changes_no_first = HUB_Ln_Changes[1:, :]
            HUB_Correlation_Matrix = np.corrcoef(HUB_Ln_Changes_no_first.T)
            Corr60 = np.asarray(HUB_Correlation_Matrix, dtype=float)
            
            logging.debug(f"Hub Curve ID for derived calc: {hub_curve_id}") # <<< ADDED
            logging.debug(f"Use Basis for derived calc: {use_basis}") # <<< ADDED
            logging.debug(f"Correlation Matrix (Corr60) shape: {Corr60.shape}") # <<< ADDED
            
            # Volatility per bucket
            vol_full = np.std(HUB_Ln_Changes_no_first, axis=0, ddof=1)
            
            # Latest curve date data
            latest_curve_date = pd.to_datetime(price_data['CurveDate'].max())
            snap = price_data[price_data['CurveDate'] == latest_curve_date].copy()
            
            # We'll save snap DataFrame separately as CSV
            snap_path = os.path.join(self.cache_dir, f"latest_values_{hub_curve_id}_{use_basis}.csv")
            snap.to_csv(snap_path, index=False)
            
            # Convert dates to strings for JSON serialization
            dates_str = [d.strftime("%Y-%m-%d") if isinstance(d, pd.Timestamp) else str(d) for d in dates]
            
            derived_data = {
                'HUB_Ln_Changes': HUB_Ln_Changes,
                'dates': dates_str,
                'seqs': seqs,
                'Corr60': Corr60,
                'vol_full': vol_full,
                'latest_curve_date': latest_curve_date.strftime("%Y-%m-%d"),
                'snap_file': os.path.basename(snap_path),
                'calculation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Cache the derived data as JSON
            with open(cache_path, 'w') as f:
                json.dump(derived_data, f, cls=NumpyJSONEncoder, indent=2)
                
            logging.info(f"Derived calculations completed in {time.time() - start_time:.2f} seconds") # <<< MODIFIED
            logging.info(f"Derived data cached to {cache_path}") # <<< MODIFIED
            logging.info(f"Latest values saved to {snap_path}") # <<< MODIFIED
            
            # Return with proper types
            derived_data['dates'] = dates
            derived_data['latest_curve_date'] = latest_curve_date
            derived_data['latest_values'] = snap
            return derived_data
        else:
            # Load from JSON cache
            try:
                logging.info(f"Loading cached derived calculations from {cache_path}...") # <<< MODIFIED
                
                with open(cache_path, 'r') as f:
                    derived_data = json.load(f, object_hook=numpy_json_decoder)
                
                # Load latest values from CSV
                snap_path = os.path.join(self.cache_dir, derived_data['snap_file'])
                snap = pd.read_csv(snap_path)
                
                # Convert date strings back to datetime objects
                derived_data['dates'] = [pd.Timestamp(d) for d in derived_data['dates']]
                derived_data['latest_curve_date'] = pd.Timestamp(derived_data['latest_curve_date'])
                derived_data['latest_values'] = snap
                
                logging.info(f"Loaded derived calculations from {cache_path}") # <<< MODIFIED
                return derived_data
                
            except Exception as e:
                logging.error(f"Error loading derived cache, recalculating: {e}") # <<< MODIFIED
                return self.get_derived_calculations(price_data, build_matrices_func, hub_curve_id, use_basis, recalculate=True)

# Initialize the global cache manager
data_cache = MonthlyDataCache()

# --------------------------
# DB Connection with Caching (Entrade DB)
# --------------------------
@lru_cache(maxsize=1)
def get_db_engine():
    """Cached database connection to avoid repeated connection setup"""
    username = keyring.get_password("EntradeDB", "username")
    password = keyring.get_password("EntradeDB", "password")
    server = 'entradev6dbprod.database.windows.net'
    database = 'EntradeMPGPROD'
    conn_str = (
        f"mssql+pyodbc://{username}:{password}@{server}:1433/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&Encrypt=yes"
        "&TrustServerCertificate=no"
    )
    return create_engine(conn_str)

# --------------------------
# NEW: Inputs DB Engine & Fetcher
# --------------------------
@lru_cache(maxsize=1)
def get_inputs_db_engine(service_name="EntradeDev"):
    """
    Create engine for the inputs DB using keyring entries.
    Expects keyring service with:
      - username (keyring.get_password(service_name, "username"))
      - password (keyring.get_password(service_name, "password"))
      - server   (keyring.get_password(service_name, "server"))  -- optional; fallback below
      - database (keyring.get_password(service_name, "database"))-- optional; fallback below

    If server/database are not in keyring, default placeholders are used and you'll
    need to change them here or store them in keyring.
    """
    username = keyring.get_password("EntradeDev", "username")
    password = keyring.get_password("EntradeDev", "password")
    server = 'entradev6db.database.windows.net'
    database = 'MPGAPIDB'
    if username is None or password is None:
        raise RuntimeError(f"Keyring entries for {service_name} are missing. Please add them.")
    conn_str = (
        f"mssql+pyodbc://{username}:{password}@{server}:1433/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&Encrypt=yes"
        "&TrustServerCertificate=no"
    )
    return create_engine(conn_str)

def get_load_profiles_to_calculate(service_name="EntradeDev", table_name="Credit_Exposure_Calc"):
    """
    Query the database for all distinct Load_Profile_IDs that have at least one record
    with a Python_Calc_Flag of NULL.
    
    Returns a list of dictionaries, where each dictionary contains the necessary input
    parameters for a single Load_Profile_ID that needs calculation.
    """
    try:
        engine = get_inputs_db_engine(service_name)
        
        # This query finds all distinct Load_Profile_IDs that have any entry with Python_Calc_Flag being NULL.
        # It then returns the first valid row of data for each of those Load_Profile_IDs.
        query = f"""
        WITH ProfilesToCalculate AS (
            SELECT DISTINCT Load_Profile_ID
            FROM {table_name}
            WHERE Python_Calc_Flag IS NULL OR Python_Calc_Flag = 0
        ),
        RankedProfiles AS (
            SELECT 
                cec.Contract_Start_Date,
                cec.Billing_Type,
                cec.Utility_State,
                cec.Term_Options,
                cec.Load_Profile_ID,
                cec.Jan_Dth, cec.Feb_Dth, cec.Mar_Dth, cec.Apr_Dth, cec.May_Dth, cec.Jun_Dth, 
                cec.Jul_Dth, cec.Aug_Dth, cec.Sep_Dth, cec.Oct_Dth, cec.Nov_Dth, cec.Dec_Dth,
                ROW_NUMBER() OVER(PARTITION BY cec.Load_Profile_ID ORDER BY cec.Contract_Start_Date) as rn
            FROM {table_name} cec
            JOIN ProfilesToCalculate ptc ON cec.Load_Profile_ID = ptc.Load_Profile_ID
        )
        SELECT 
            Contract_Start_Date,
            Billing_Type,
            Utility_State,
            Term_Options,
            Load_Profile_ID,
            Jan_Dth, Feb_Dth, Mar_Dth, Apr_Dth, May_Dth, Jun_Dth, 
            Jul_Dth, Aug_Dth, Sep_Dth, Oct_Dth, Nov_Dth, Dec_Dth
        FROM RankedProfiles
        WHERE rn = 1
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logging.info("No Load Profiles need calculation at this time.") # <<< MODIFIED
            return []

        profiles_to_process = []
        for index, row in df.iterrows():
            profile_data = {}
            # Parse simple types
            if pd.notna(row.get("Contract_Start_Date")):
                dt = pd.to_datetime(row["Contract_Start_Date"], errors="coerce")
                profile_data["start_month_input"] = int(dt.month) if pd.notna(dt) else 1
            
            if pd.notna(row.get("Load_Profile_ID")):
                profile_data["Load_Profile_ID"] = str(row["Load_Profile_ID"])
        
            if pd.notna(row.get("Billing_Type")):
                profile_data["Billing_Type"] = str(row["Billing_Type"])
        
            if pd.notna(row.get("Utility_State")):
                profile_data["state"] = str(row["Utility_State"])

            # Parse Term_Options
            if pd.notna(row.get("Term_Options")):
                try:
                    parsed = json.loads(row["Term_Options"])
                    if isinstance(parsed, list):
                        profile_data["terms"] = [int(x) for x in parsed]
                except Exception:
                    # try comma-separated fallback
                    s = str(row["Term_Options"])
                    parts = [p.strip() for p in s.strip("[]").split(",") if p.strip()]
                    profile_data["terms"] = [int(x) for x in parts]

            # Build Monthly_Volume_12Months from individual month columns
            month_columns = ["Jan_Dth", "Feb_Dth", "Mar_Dth", "Apr_Dth", "May_Dth", "Jun_Dth", 
                             "Jul_Dth", "Aug_Dth", "Sep_Dth", "Oct_Dth", "Nov_Dth", "Dec_Dth"]
        
            monthly_volumes = []
            for col in month_columns:
                val = row.get(col, 0)  # default to 0 if column missing
                monthly_volumes.append(int(val) if pd.notna(val) else 0)
        
            profile_data["Monthly_Volume_12Months"] = monthly_volumes
            
            profiles_to_process.append(profile_data)

        logging.info(f"Found {len(profiles_to_process)} Load Profiles that need calculation.") # <<< MODIFIED
        return profiles_to_process

    except Exception as e:
        logging.error(f"Error fetching load profiles to calculate: {e}") # <<< MODIFIED
        return []

def check_calculation_flags_for_profile(load_profile_id, all_terms, service_name="EntradeDev", table_name="Credit_Exposure_Calc"):
    """
    For a given Load_Profile_ID, check which terms need calculation.
    """
    try:
        engine = get_inputs_db_engine(service_name)
        
        # Get the status for all terms of a given Load Profile ID in one go
        query = f"""
        SELECT Term_Split, Python_Calc_Flag
        FROM {table_name}
        WHERE Load_Profile_ID = :load_profile_id
        """
        
        df = pd.read_sql(query, engine, params={'load_profile_id': load_profile_id})
        
        # Create a dictionary of term -> calc_flag
        existing_statuses = df.set_index('Term_Split')['Python_Calc_Flag'].to_dict()
        
        terms_to_calculate = []
        for term in all_terms:
            status = existing_statuses.get(term)
            if status is None or status == 0:
                terms_to_calculate.append(term)
                logging.info(f"  - Term {term}: Needs calculation (Flag is {status})") # <<< MODIFIED
            else:
                logging.info(f"  - Term {term}: Already calculated (Flag is 1)") # <<< MODIFIED

        return terms_to_calculate

    except Exception as e:
        logging.error(f"Error checking calculation flags for Load Profile ID {load_profile_id}: {e}") # <<< MODIFIED
        # On error, assume all terms need calculation as a fallback
        return all_terms
    
def update_calculation_flag_for_profile(load_profile_id, service_name="EntradeDev", table_name="Credit_Exposure_Calc"):
    """
    Update Python_Calc_Flag to 1 for all records associated with a given Load_Profile_ID.
    """
    try:
        engine = get_inputs_db_engine(service_name)
        
        update_query = f"""
        UPDATE {table_name}
        SET Python_Calc_Flag = 1
        WHERE Load_Profile_ID = :load_profile_id
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(update_query), {'load_profile_id': load_profile_id})
            conn.commit()
            
            if result.rowcount > 0:
                logging.info(f"✅ Updated Python_Calc_Flag to 1 for {result.rowcount} records of Load Profile ID {load_profile_id}") # <<< MODIFIED
                return True
            else:
                logging.warning(f"⚠️ No records updated for Load Profile ID {load_profile_id}. This might be unexpected.") # <<< MODIFIED
                return False
        
    except Exception as e:
        logging.error(f"Error updating calculation flag for Load Profile ID {load_profile_id}: {e}") # <<< MODIFIED
        return False

def create_var_results_table(service_name="EntradeDev", table_name="Credit_Exposure_Calc"):
    """
    Create the Credit_Exposure_Calc table if it doesn't exist (with updated columns)
    
    Parameters:
    service_name (str): Keyring service name for database connection
    table_name (str): Target table name in SQL
    """
    try:
        engine = get_inputs_db_engine(service_name)
        
        # Check if table exists and if it has the required columns
        check_table_sql = f"""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}')
        BEGIN
            -- Table doesn't exist, create it
            CREATE TABLE {table_name} (
                ID INT IDENTITY(1,1) PRIMARY KEY,
                Deal_ID NVARCHAR(50),
                Contract_Start_Date DATETIME2,
                Term_Split INT NOT NULL,
                Load_Profile_ID NVARCHAR(50),
                Billing_Type NVARCHAR(100),
                Utility_State NVARCHAR(50),
                Term_Options NVARCHAR(MAX),
                Jan_Dth INT,
                Feb_Dth INT,
                Mar_Dth INT,
                Apr_Dth INT,
                May_Dth INT,
                Jun_Dth INT,
                Jul_Dth INT,
                Aug_Dth INT,
                Sep_Dth INT,
                Oct_Dth INT,
                Nov_Dth INT,
                Dec_Dth INT,
                PFE DECIMAL(18,2),
                MTM DECIMAL(18,2),
                Fixed_Price_AR DECIMAL(18,2),
                Variable_Price_AR DECIMAL(18,2),
                Hub NVARCHAR(50),
                Calculation_Date DATETIME2,
                Flat_Gas_Price DECIMAL(18,2),
                All_In_Price DECIMAL(18,2),
                Python_Calc_Flag INT,
                Created_Date DATETIME2 DEFAULT GETDATE()
            )
        END
        ELSE
        BEGIN
            -- Table exists, check and add missing columns
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'PFE')
                ALTER TABLE {table_name} ADD PFE DECIMAL(18,2)
            
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'MTM')
                ALTER TABLE {table_name} ADD MTM DECIMAL(18,2)
            
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'Fixed_Price_AR')
                ALTER TABLE {table_name} ADD Fixed_Price_AR DECIMAL(18,2)
            
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'Variable_Price_AR')
                ALTER TABLE {table_name} ADD Variable_Price_AR DECIMAL(18,2)
            
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'Hub')
                ALTER TABLE {table_name} ADD Hub NVARCHAR(50)
            
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'Calculation_Date')
                ALTER TABLE {table_name} ADD Calculation_Date DATETIME2
            
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'Flat_Gas_Price')
                ALTER TABLE {table_name} ADD Flat_Gas_Price DECIMAL(18,2)
            
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'All_In_Price')
                ALTER TABLE {table_name} ADD All_In_Price DECIMAL(18,2)
            
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'Python_Calc_Flag')
                ALTER TABLE {table_name} ADD Python_Calc_Flag INT
        END
        """
        
        with engine.connect() as conn:
            conn.execute(text(check_table_sql))
            conn.commit()
        
        logging.info(f"Table {table_name} created or updated with required columns") # <<< MODIFIED
        return True
        
    except Exception as e:
        logging.error(f"Error creating/updating table: {e}") # <<< MODIFIED
        return False

def export_results_to_sql(results, load_profile_id, service_name="EntradeDev", table_name="Credit_Exposure_Calc"):
    """
    Export VAR analysis results to SQL database for a specific Load_Profile_ID.
    It updates all records for that Load_Profile_ID.
    """
    try:
        engine = get_inputs_db_engine(service_name)
        current_time = datetime.now()
        
        with engine.connect() as conn:
            for term, term_results in results.items():
                update_query = f"""
                UPDATE {table_name}
                SET PFE = :pfe,
                    MTM = :mtm,
                    Fixed_Price_AR = :fixed_price_ar,
                    Variable_Price_AR = :variable_price_ar,
                    Hub = :hub,
                    Calculation_Date = :calc_date,
                    Flat_Gas_Price = :flat_gas_price,
                    All_In_Price = :all_in_price
                WHERE Load_Profile_ID = :load_profile_id AND Term_Split = :term
                """
                
                conn.execute(text(update_query), {
                    'pfe': float(term_results['PFE']),
                    'mtm': float(term_results['MTM_at_max']),
                    'fixed_price_ar': float(term_results['Fixed_Price_AR_max']),
                    'variable_price_ar': float(term_results['Max_Var_Price']),
                    'hub': term_results['Hub'],
                    'calc_date': current_time,
                    'flat_gas_price': float(term_results['Flat_Gas_Price']),
                    'all_in_price': float(term_results['All_In_Price']),
                    'load_profile_id': load_profile_id,
                    'term': term
                })

            conn.commit()
            logging.info(f"Successfully exported results for Load Profile ID: {load_profile_id}") # <<< MODIFIED

        return True
        
    except Exception as e:
        logging.error(f"Error exporting to SQL for Load Profile ID {load_profile_id}: {e}") # <<< MODIFIED
        import traceback
        traceback.print_exc()
        return False

def query_var_results(service_name="EntradeDev", table_name="Credit_Exposure_Calc", days_back=30):
    """
    Query recent VAR results from SQL database
    
    Parameters:
    service_name (str): Keyring service name for database connection
    table_name (str): Source table name in SQL
    days_back (int): Number of days back to query
    
    Returns:
    DataFrame: Recent VAR results
    """
    try:
        engine = get_inputs_db_engine(service_name)
        
        query = f"""
        SELECT *
        FROM {table_name}
        WHERE Calculation_Date >= DATEADD(day, -{days_back}, GETDATE())
        ORDER BY Calculation_Date DESC, Term_Split
        """
        
        df = pd.read_sql(query, engine)
        logging.info(f"Retrieved {len(df)} VAR result records from the last {days_back} days") # <<< MODIFIED
        return df
        
    except Exception as e:
        logging.error(f"Error querying VAR results: {e}") # <<< MODIFIED
        return pd.DataFrame()

# --------------------------
# Price Data Fetcher (unchanged)
# --------------------------
def get_HH_price_data(hub_curve_id, use_basis):
    """
    Get price data for specified hub
    
    Parameters:
    hub_curve_id (str): The curve ID for the hub
    use_basis (bool): If True, add the basis curve to HH; if False, use hub curve directly
    
    Returns:
    DataFrame: Price data
    """
    engine = get_db_engine()
    start_time = time.time()
    
    # Keep the original SQL query to ensure results match
    if use_basis and hub_curve_id != "30000":
        # MODIFIED SQL QUERY THAT COMBINES HUB AND BASIS PRICES
 # MODIFIED SQL QUERY THAT COMBINES HUB AND BASIS PRICES
        query = f"""
WITH HH AS (
    SELECT CurveDate, RangeSEQ, Value, EstimateFlag, CurveUID, SandboxUID
    FROM EntradeMPGProd.enstep.CurveRangeValues
    WHERE RangeSEQ <= 61
      AND CurveUID = '30000'
      AND CurveDate BETWEEN DATEADD(DAY, -1, DATEADD(YEAR, -1, EOMONTH(GETDATE(), -1)))
                       AND EOMONTH(GETDATE(), -1)
),
BASIS AS (
    SELECT CurveDate, RangeSEQ, Value
    FROM EntradeMPGProd.enstep.CurveRangeValues
    WHERE RangeSEQ <= 61
      AND CurveUID = '{hub_curve_id}'  -- (Adjust this to the actual basis curve if different)
      AND CurveDate BETWEEN DATEADD(DAY, -1, DATEADD(YEAR, -1, EOMONTH(GETDATE(), -1)))
                       AND EOMONTH(GETDATE(), -1)
),
BaseData AS (
    SELECT
        HH.CurveUID,
        HH.SandboxUID,
        HH.CurveDate,
        HH.RangeSEQ,
        HH.Value + COALESCE(BASIS.Value, 0) AS Value,
        HH.EstimateFlag,
        ROW_NUMBER() OVER (PARTITION BY HH.CurveUID, HH.CurveDate ORDER BY HH.RangeSEQ) AS NewRangeSEQ,
        DENSE_RANK() OVER (
            PARTITION BY HH.CurveUID, YEAR(HH.CurveDate), MONTH(HH.CurveDate)
            ORDER BY HH.CurveDate DESC
        ) AS DateRank
    FROM HH
    LEFT JOIN BASIS
      ON BASIS.CurveDate = HH.CurveDate
     AND BASIS.RangeSEQ = HH.RangeSEQ
),
FilteredData AS (
    SELECT *
    FROM BaseData
    WHERE NewRangeSEQ <= 61
),
WindowedData AS (
    SELECT
        fd.*,
        -- Special denominator using (RangeSEQ + 1) from the immediately prior curve date
        (SELECT prev_data.Value
           FROM FilteredData prev_data
          WHERE prev_data.CurveUID = fd.CurveUID
            AND prev_data.CurveDate = (
                SELECT MAX(sub_data.CurveDate)
                  FROM FilteredData sub_data
                 WHERE sub_data.CurveUID = fd.CurveUID
                   AND sub_data.CurveDate < fd.CurveDate
            )
            AND prev_data.NewRangeSEQ = fd.NewRangeSEQ + 1
        ) AS TwoLastDenomValue,
        -- Normal previous day same bucket
        LAG(Value) OVER (PARTITION BY CurveUID, NewRangeSEQ ORDER BY CurveDate) AS PrevDateValue
    FROM FilteredData fd
),
Computed AS (
    SELECT
        CurveUID,
        SandboxUID,
        CurveDate,
        RangeSEQ,
        Value,
        EstimateFlag,
        NewRangeSEQ,
        DateRank,
        CASE WHEN DateRank = 2 THEN 1 ELSE 0 END AS IsSecondLargestDate,
        CASE WHEN DateRank = 2 THEN TwoLastDenomValue ELSE NULL END AS TwoLast_Denominator,
        CASE WHEN DateRank <> 2 THEN PrevDateValue ELSE NULL END AS Normal_Denominator,
        -- Base LN calc before special bucket 60 override
        CASE
            WHEN DateRank = 2 THEN LOG(Value / NULLIF(TwoLastDenomValue, 0))
            WHEN PrevDateValue > 0 THEN LOG(Value / NULLIF(PrevDateValue, 0))
            ELSE 0
        END AS LN_Calc_Base,
        -- Capture bucket 59 LN_Calc_Base for reuse on bucket 60 (same CurveDate)
        MAX(
            CASE WHEN NewRangeSEQ = 59 THEN
                CASE
                    WHEN DateRank = 2 THEN LOG(Value / NULLIF(TwoLastDenomValue, 0))
                    WHEN PrevDateValue > 0 THEN LOG(Value / NULLIF(PrevDateValue, 0))
                    ELSE 0
                END
            END
        ) OVER (PARTITION BY CurveUID, CurveDate) AS LN_Calc_59_For_Copy
    FROM WindowedData
)
SELECT
    CurveUID,
    SandboxUID,
    CurveDate,
    RangeSEQ,
    Value,
    EstimateFlag,
    NewRangeSEQ,
    DateRank,
    IsSecondLargestDate,
    TwoLast_Denominator,
    Normal_Denominator,
    CASE
        WHEN DateRank = 2 AND NewRangeSEQ = 60 THEN LN_Calc_59_For_Copy
        ELSE LN_Calc_Base
    END AS LN_Calc
FROM Computed
WHERE NewRangeSEQ <= 61
ORDER BY CurveUID, CurveDate, NewRangeSEQ;
"""
    else:
        # Use direct hub data without combining
        query = f"""
WITH BaseData AS (
    SELECT
        CurveUID,
        SandboxUID,
        CurveDate,
        RangeSEQ,
        Value,
        EstimateFlag,
        ROW_NUMBER() OVER (PARTITION BY CurveUID, CurveDate ORDER BY RangeSEQ) AS NewRangeSEQ,
        DENSE_RANK() OVER (
            PARTITION BY CurveUID, YEAR(CurveDate), MONTH(CurveDate)
            ORDER BY CurveDate DESC
        ) AS DateRank
    FROM EntradeMPGProd.enstep.CurveRangeValues
    WHERE RangeSEQ <= 61
      AND CurveUID = '{hub_curve_id}'
      AND CurveDate BETWEEN DATEADD(DAY, -1, DATEADD(YEAR, -1, EOMONTH(GETDATE(), -1)))
                       AND EOMONTH(GETDATE(), -1)
),
FilteredData AS (
    SELECT *
    FROM BaseData
    WHERE NewRangeSEQ <= 61
),
WindowedData AS (
    SELECT
        fd.*,
        (SELECT prev_data.Value
           FROM FilteredData prev_data
          WHERE prev_data.CurveUID = fd.CurveUID
            AND prev_data.CurveDate = (
                SELECT MAX(sub_data.CurveDate)
                  FROM FilteredData sub_data
                 WHERE sub_data.CurveUID = fd.CurveUID
                   AND sub_data.CurveDate < fd.CurveDate
            )
            AND prev_data.NewRangeSEQ = fd.NewRangeSEQ + 1
        ) AS TwoLastDenomValue,
        LAG(Value) OVER (PARTITION BY CurveUID, NewRangeSEQ ORDER BY CurveDate) AS PrevDateValue
    FROM FilteredData fd
),
Computed AS (
    SELECT
        CurveUID,
        SandboxUID,
        CurveDate,
        RangeSEQ,
        Value,
        EstimateFlag,
        NewRangeSEQ,
        DateRank,
        CASE WHEN DateRank = 2 THEN 1 ELSE 0 END AS IsSecondLargestDate,
        CASE WHEN DateRank = 2 THEN TwoLastDenomValue ELSE NULL END AS TwoLast_Denominator,
        CASE WHEN DateRank <> 2 THEN PrevDateValue ELSE NULL END AS Normal_Denominator,
        -- Base LN calc (before special copy rule)
        CASE
            WHEN DateRank = 2 THEN LOG(Value / NULLIF(TwoLastDenomValue, 0))
            WHEN PrevDateValue > 0 THEN LOG(Value / NULLIF(PrevDateValue, 0))
            ELSE 0
        END AS LN_Calc_Base,
        -- Capture the bucket 59 LN_Calc_Base for possible reuse
        MAX(
            CASE WHEN NewRangeSEQ = 59 THEN
                CASE
                    WHEN DateRank = 2 THEN LOG(Value / NULLIF(TwoLastDenomValue, 0))
                    WHEN PrevDateValue > 0 THEN LOG(Value / NULLIF(PrevDateValue, 0))
                    ELSE 0
                END
            END
        ) OVER (PARTITION BY CurveUID, CurveDate) AS LN_Calc_59_For_Copy
    FROM WindowedData
)
SELECT
    CurveUID,
    SandboxUID,
    CurveDate,
    RangeSEQ,
    Value,
    EstimateFlag,
    NewRangeSEQ,
    DateRank,
    IsSecondLargestDate,
    TwoLast_Denominator,
    Normal_Denominator,
    CASE
        WHEN DateRank = 2 AND NewRangeSEQ = 60 THEN LN_Calc_59_For_Copy
        ELSE LN_Calc_Base
    END AS LN_Calc
FROM Computed
WHERE NewRangeSEQ <= 61
ORDER BY CurveUID, CurveDate, NewRangeSEQ;
"""
    
    df = pd.read_sql(query, engine)
    df['CurveDate'] = pd.to_datetime(df['CurveDate'])
    print(f"SQL query executed in {time.time() - start_time:.2f} seconds")
    
    return df

def _copy_58_to_59_on_second_largest(df, value_col='LN_Calc'):
    """
    For each CurveDate (and CurveUID if present), when IsSecondLargestDate == 1,
    set LN_Calc at NewRangeSEQ=59 equal to the value at NewRangeSEQ=58.
    """
    key_cols = ['CurveUID', 'CurveDate'] if 'CurveUID' in df.columns else ['CurveDate']

    # Get LN_Calc at bucket 58
    v59 = (df.loc[df['NewRangeSEQ'] == 59, key_cols + [value_col]]
             .rename(columns={value_col: '_ln59'}))

    # Attach to all rows and overwrite bucket 59 where needed
    df = df.merge(v59, on=key_cols, how='left')
    mask = (df['IsSecondLargestDate'] == 1) & (df['NewRangeSEQ'] == 60) & df['_ln59'].notna()
    df.loc[mask, value_col] = df.loc[mask, '_ln59'].values

    return df.drop(columns=['_ln59'])

def build_price_matrix(df):
    df = df.copy()

    # 58 -> 59 on second-largest date
    df = _copy_58_to_59_on_second_largest(df, value_col='LN_Calc')

    # Remove/ignore bucket 60 completely
    df = df[df['NewRangeSEQ'] <= 60]

    # Pivot, keep columns 1..59 in order
    pivot = df.pivot(index='CurveDate', columns='NewRangeSEQ', values='LN_Calc')
    pivot = pivot.sort_index().sort_index(axis=1).reindex(columns=list(range(1, 61)))

    matrix = pivot.fillna(0).to_numpy()
    return matrix, pivot.index.to_list(), pivot.columns.to_list()

def month_name(n):
    return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][n-1]

def var_single_day_from_DV(DV60, Corr60):
    val = DV60 @ Corr60 @ DV60
    return float(np.sqrt(val if val > 0.0 else 0.0))

def shift_with_leading_zero(x, n):
    a = np.asarray(x, dtype=float).ravel()
    a = np.concatenate(([0.0], a))
    if a.size < n:
        a = np.pad(a, (0, n - a.size), mode='constant')
    else:
        a = a[:n]
    return a

def calculate_var_metrics(state_code, Term_length, start_month_input, Billing_Type, Monthly_Volume_12Months):
    """
    Calculate VAR metrics for a specific state or hub
    Modified to use monthly caching for SQL data and derived calculations
    
    Parameters:
    state_code (str): Two-letter state code to determine hub
    Term_length (int): Contract term length (12 or 24 months)
    start_month_input (int): Start month of contract (1-12)
    Billing_Type (str): "Supplier" or other
    Monthly_Volume_12Months (list): List of 12 monthly volumes
    
    Returns:
    dict: VAR analysis results
    """
    start_time = time.time()
    
    hub = STATE_TO_HUB[state_code]
    logging.info(f"HUB: {hub}") # <<< MODIFIED
    
    # Get the curve ID for the selected hub
    hub_curve_id = HUB_CURVE_IDS.get(hub.lower(), "30000") # CHECK THIS
    
    logging.info(f"Using hub: {hub.upper()} with curve ID: {hub_curve_id}") # <<< MODIFIED
    
    # Use basis calculation for non-HH hubs, direct data for HH
    use_basis = (hub.lower() != "hub")
    
    # Get price data using the cache manager
    hh_data = data_cache.get_price_data(get_HH_price_data, hub_curve_id, use_basis)
    
    # Get derived calculations using the cache manager
    derived_data = data_cache.get_derived_calculations(hh_data, build_price_matrix, hub_curve_id, use_basis)
    
    # Extract cached values
    HUB_Ln_Changes = derived_data['HUB_Ln_Changes']
    dates = derived_data['dates']
    Corr60 = derived_data['Corr60']
    vol_full = derived_data['vol_full']
    latest_curve_date = derived_data['latest_curve_date']
    snap = derived_data['latest_values']
    
    print("latest vals",snap)
    
    # <<< ADDED: Log the 'snap' dataframe which contains the latest prices
    logging.info(f"Latest prices snapshot ('snap') from {latest_curve_date.date()}:\n{snap.head()}")

    # Prepare volume data
    Monthly_Contract_Volume_Therms = []
    for i in range(Term_length):
        Monthly_Contract_Volume_Therms.append(Monthly_Volume_12Months[i % 12])
    
    months = np.arange(1, Term_length+1, dtype=np.float64).reshape(-1, 1)
    volumes = np.array(Monthly_Contract_Volume_Therms, dtype=np.float64).reshape(-1, 1)
    volumes_divided = volumes
    Total_Contract_Volume = np.hstack((months, volumes_divided))
    SF = 1.75 if Billing_Type == "Supplier Consolidated" else 1
    
    # Lane months calculation
    current_month = int(datetime.now().month)
    first_month = (current_month % 12) + 1
    months_before_start = (start_month_input - first_month + 12) % 12
    lane_len = months_before_start + Term_length
    LANE_MONTHS = [((first_month + i - 1) % 12) + 1 for i in range(lane_len)]
    
    # <<< ADDED: Log lane month details
    logging.info(f"Lane details: Start Month={start_month_input}, Months before start={months_before_start}, Term={Term_length}, Total Lane Length={lane_len}")
    logging.info(f"LANE_MONTHS: {LANE_MONTHS}")
    
    # Volatility per bucket for lane
    Volatility_lane = vol_full[:len(LANE_MONTHS)]
    
    # Prices (one per lane bucket)
    seq_indexer = snap.set_index('NewRangeSEQ')
    
    try:
        price_lane = seq_indexer.loc[list(range(1, len(LANE_MONTHS)+1)), 'Value'].to_numpy()
    except:
        # Handle missing indices safely
        price_lane = np.zeros(len(LANE_MONTHS))
        for i, idx in enumerate(range(1, len(LANE_MONTHS)+1)):
            try:
                price_lane[i] = seq_indexer.loc[idx, 'Value']
            except:
                if i > 0:
                    price_lane[i] = price_lane[i-1]  # Use previous value if available
    
    # <<< ADDED: This is the critical logging point for your suspicion
    logging.info(f"Prices for forward months (price_lane): \n{price_lane}")

    # Calendar-month → volume map (ALWAYS calendar 1..12)
    calendar_month_to_vol = {m: float(Monthly_Volume_12Months[m-1]) for m in range(1, 13)}

    # Map every lane bucket to its calendar-month volume
    vol_lane = np.array([calendar_month_to_vol[m] for m in LANE_MONTHS], dtype=float)

    logging.info(f"vol_lane: \n{vol_lane}")

    Investment_lane = price_lane * vol_lane
    
    # <<< ADDED: Log the investment lane
    logging.info(f"Investment Lane (price * volume): \n{Investment_lane}")
    
    # --------------------------
    # DV Row Builder Function
    # --------------------------
    def build_DV_row_month1_forwardN(forward_month, start_month_input, month, Investment_lane, Volatility_lane):
        win_len = int(forward_month)
        # anchor relative to the same lane start (first_month)
        base_abs = ((start_month_input - first_month + 12) % 12) + (month - 1)
        
        L = len(LANE_MONTHS)
        DV_lane = np.zeros(L, dtype=float)
        
        for idx in range(Term_length):
            j_rel = base_abs + idx
            if j_rel >= L:
                break
            if (j_rel + 1) < win_len:
                continue
                
            left = j_rel - (win_len - 1)
            if left < 0 or left > L - 1:
                continue
                
            v_slice = Volatility_lane[left : j_rel + 1]
            v_slice = v_slice[np.isfinite(v_slice)]
            v_slice = v_slice[v_slice > 0]
            gmean = float(np.exp(np.log(v_slice).mean())) if v_slice.size else 0.0
            
            exposure = float(Investment_lane[j_rel]) * gmean
            DV_lane[left] = exposure
        
        DV60 = np.zeros(60, dtype=float)
        DV60[:L] = DV_lane
        return DV60
    
    # --------------------------
    # VAR Single Day & VAR Monthly Calculation
    # --------------------------
    month = 1
    rows = []
    
    # Pre-allocate array to store adjusted VAR values
    adjvar_per_lane = np.zeros(len(LANE_MONTHS))
    
    for forward_from_sep in range(1, len(LANE_MONTHS) + 1):
        DV = build_DV_row_month1_forwardN(
            forward_month=forward_from_sep,
            start_month_input=start_month_input,
            month=month,
            Investment_lane=Investment_lane,
            Volatility_lane=Volatility_lane
        )
        vsd = var_single_day_from_DV(DV, Corr60)
        vmonthly = float(vsd * np.sqrt(252.0 * forward_from_sep / 12.0))
        
        k = forward_from_sep - 1
        denom_count = min(Term_length, len(LANE_MONTHS) - k)
        
        # Safe slicing with bounds check
        if k >= len(vol_lane):
            denom_vol = 0.0
        elif k + denom_count > len(vol_lane):
            denom_vol = float(vol_lane[k:].sum())
        else:
            denom_vol = float(vol_lane[k : k + denom_count].sum())
            
        adj_var = (vmonthly / denom_vol) if denom_vol > 0 else 0.0
        
        # Store for later use
        if k < len(adjvar_per_lane):
            adjvar_per_lane[k] = adj_var
        
        mnum = LANE_MONTHS[k] if k < len(LANE_MONTHS) else 1
        current_year = datetime.now().year
        month_offset = k
        year_offset = month_offset // 12
        year = current_year + year_offset
        if (current_month + month_offset) % 12 < current_month:
            year += 1
            
        label = f"{month_name(mnum)}-{year}"
        
        rows.append({
            "month": label,
            "forward_from_sep": forward_from_sep,
            "Denom_Volume": denom_vol,
            "VAR_Single_Day": vsd,
            "VAR_Monthly": vmonthly,
            "Adj_VAR": adj_var
        })
    
    VAR_table = pd.DataFrame(rows)
    # <<< ADDED: Log the VAR table
    # Use pd.option_context to display all rows without truncation
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        logging.info(f"Calculated VAR_table:\n{VAR_table}")

    # --------------------------
    # Monthly Variable Price Exposure
    # --------------------------
    deliver_idx_start = ((start_month_input - first_month + 12) % 12)
    deliver_idx_end   = deliver_idx_start + Term_length
    idx_term = [i for i in range(deliver_idx_start, deliver_idx_end) if i < len(LANE_MONTHS)]
    
    if len(idx_term) > 0:
        forward_prices_term = price_lane[idx_term]
        daily_ln_sigma_term = Volatility_lane[idx_term]
        monthly_scale = np.sqrt(21.0)
        
        # Volumes for the delivery months from calendar-month map
        vol_vec_term = np.array([calendar_month_to_vol[LANE_MONTHS[i]] for i in idx_term], dtype=float)
        
        # adj_var per lane (already lane-aligned)
        adjvar_term = adjvar_per_lane[idx_term]
        
        Monthly_Variable_Price_Exposure_adj = (forward_prices_term + SF + 1.645 * adjvar_term) * vol_vec_term
        Monthly_Variable_Price_Exposure_adj = np.nan_to_num(Monthly_Variable_Price_Exposure_adj, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        Monthly_Variable_Price_Exposure_adj = np.array([])
    
    # --------------------------
    # Weighted Average Strip Price & Exposure Calculations
    # --------------------------
    Monthly_Net_Position = vol_lane[deliver_idx_start:min(deliver_idx_end, len(vol_lane))].astype(float)
    Monthly_Price        = price_lane[deliver_idx_start:min(deliver_idx_end, len(price_lane))].astype(float)
    
    price_lane_float = price_lane.astype(float)
    notional = vol_lane.astype(float)
    
    tail_w   = np.flip(np.cumsum(np.flip(notional)))
    tail_sum = np.flip(np.cumsum(np.flip(price_lane_float * notional)))
    
    # Safe division
    Weighted_Avg_Strip_Price = np.zeros_like(tail_sum)
    mask = tail_w != 0
    Weighted_Avg_Strip_Price[mask] = tail_sum[mask] / tail_w[mask]
    
    # Get flat gas price safely
    if deliver_idx_start < len(Weighted_Avg_Strip_Price):
        Flat_Gas_Price = float(Weighted_Avg_Strip_Price[deliver_idx_start])
    else:
        Flat_Gas_Price = float(Weighted_Avg_Strip_Price[-1]) if len(Weighted_Avg_Strip_Price) > 0 else 0.0
        
    All_In_Price = Flat_Gas_Price + SF
    All_In_Price_vec = np.full_like(Monthly_Net_Position, All_In_Price, dtype=float)
    
    # Fixed Price AR — current + prior month
    mnp = np.asarray(Monthly_Net_Position, dtype=float)
    prev_mnp = np.zeros_like(mnp)
    if len(mnp) > 1:
        prev_mnp[1:] = mnp[:-1]
    Fixed_Price_AR = All_In_Price * (mnp + prev_mnp)
    
    # Rolling MTM loop
    idx0 = deliver_idx_start + month
    idx1 = min(idx0 + Term_length, len(price_lane))
    
    price_per_month = Weighted_Avg_Strip_Price.astype(float)
    mnp_full = vol_lane.astype(float)
    
    Rolling_MTM = []
    Downward_Equivalent = []
    Potential_Risk_1sig = []
    
    for k in range(idx0, idx1):
        if k >= len(price_per_month):
            break
        flat_minus_price = float(Flat_Gas_Price) - price_per_month[k]
        tail = mnp_full[k:idx1].sum() if k < len(mnp_full) else 0
        Rolling_MTM.append(flat_minus_price * tail)
        
        if k < len(adjvar_per_lane):
            # Handle potential division by zero
            if price_per_month[k] <= 0:
                Downward_Equivalent.append(0.0)
                Potential_Risk_1sig.append(0.0)
            else:
                try:
                    downward_eq = price_per_month[k] - (price_per_month[k] * np.exp(-np.log((price_per_month[k] + adjvar_per_lane[k]) / price_per_month[k])))
                    Downward_Equivalent.append(downward_eq)
                    Potential_Risk_1sig.append(float(downward_eq) * tail)
                except:
                    # Handle any numerical issues
                    Downward_Equivalent.append(0.0)
                    Potential_Risk_1sig.append(0.0)
        else:
            Downward_Equivalent.append(0.0)
            Potential_Risk_1sig.append(0.0)
    
    Rolling_MTM = np.array(Rolling_MTM, dtype=float)
    Potential_Risk_1sig = np.array(Potential_Risk_1sig, dtype=float)
    Potential_Risk = 1.645 * Potential_Risk_1sig
    
    term_size = len(Fixed_Price_AR)
    Rolling_MTM = shift_with_leading_zero(Rolling_MTM, term_size)
    Potential_Risk = shift_with_leading_zero(Potential_Risk, term_size)
    
    Total_Exposure = Rolling_MTM + Potential_Risk + Fixed_Price_AR
    
    # <<< ADDED: Log the final exposure vector
    logging.info(f"Final Total_Exposure vector: \n{Total_Exposure}")
    
    # Find maximum exposure with safety checks
    if len(Total_Exposure) > 0:
        ind = int(np.nanargmax(Total_Exposure))
        MTM_at_max = float(Rolling_MTM[ind] + Potential_Risk[ind])
        Fixed_Price_AR_max = Fixed_Price_AR[ind]
        PFE = Total_Exposure[ind]
    else:
        MTM_at_max = 0.0
        Fixed_Price_AR_max = 0.0
        PFE = 0.0
    
    Max_Var_Price = float(np.max(Monthly_Variable_Price_Exposure_adj)) if len(Monthly_Variable_Price_Exposure_adj) > 0 else 0.0
    
    results = {
        "Term_length": Term_length,
        "Hub": hub.upper(),
        "PFE": PFE,
        "MTM_at_max": MTM_at_max,
        "Fixed_Price_AR_max": Fixed_Price_AR_max,
        "Max_Var_Price": Max_Var_Price,
        "Flat_Gas_Price": Flat_Gas_Price,
        "All_In_Price": All_In_Price,
        "VAR_table": VAR_table,
        "Monthly_Variable_Price_Exposure": Monthly_Variable_Price_Exposure_adj,
        "Total_Exposure": Total_Exposure
    }
    
    # <<< ADDED: Log the final results dictionary for this term
    # Create a copy for logging so we don't show the full (and long) VAR_table again
    results_for_log = results.copy()
    results_for_log.pop('VAR_table', None) # Remove large table from log
    logging.info(f"Final results for term {Term_length}: \n{json.dumps(results_for_log, indent=2, cls=NumpyJSONEncoder)}")
    
    logging.info(f"VAR calculation for {Term_length}-month term completed in {time.time() - start_time:.2f} seconds") # <<< MODIFIED
    return results

# --------------------------
# Run Calculation
# --------------------------
def run_multiterm_analysis(state, terms_to_run, start_month_input, Billing_Type, Monthly_Volume_12Months):
    """
    Run VAR analysis for multiple term lengths for a given set of inputs.
    """
    start_time = time.time()
    results = {}
    
    logging.info(f"Today is {datetime.now().strftime('%Y-%m-%d')}") # <<< MODIFIED
    
    # Print information about selected hub
    hub_code = STATE_TO_HUB.get(state.upper(), "hub") if state else "hub"
    logging.info(f"Running analysis for state '{state}' using {hub_code.upper()} hub") # <<< MODIFIED
        
    # Run for each term length
    for term in terms_to_run:
        logging.info(f"Running analysis for {term}-month term...") # <<< MODIFIED
        
        try:
            # <<< ADDED: Log the inputs for this specific term calculation
            logging.info(f"Inputs for calculate_var_metrics: state_code='{state}', Term_length={term}, start_month_input={start_month_input}, Billing_Type='{Billing_Type}'")
            logging.info(f"Monthly_Volume_12Months: {Monthly_Volume_12Months}")

            # Run VAR calculation for this term
            term_result = calculate_var_metrics(
                state_code=state,
                Term_length=term,
                start_month_input=start_month_input,
                Billing_Type=Billing_Type,
                Monthly_Volume_12Months=Monthly_Volume_12Months
            )
            results[term] = term_result
            
            # Display summary
            logging.info(f"\n--- {term}-Month Term Summary ---") # <<< MODIFIED
            logging.info(f"Hub: {term_result['Hub']}") # <<< MODIFIED
            logging.info(f"PFE: {term_result['PFE']:,.2f}") # <<< MODIFIED
            logging.info(f"MTM: {term_result['MTM_at_max']:,.2f}") # <<< MODIFIED
            logging.info(f"Fixed Price AR: {term_result['Fixed_Price_AR_max']:,.2f}") # <<< MODIFIED
            logging.info(f"Variable Price: {term_result['Max_Var_Price']:,.2f}") # <<< MODIFIED
            logging.info("----------------------------\n") # <<< MODIFIED
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.error(f"Error processing {term}-month term: {e}") # <<< MODIFIED
    
    logging.info(f"All analyses completed in {time.time() - start_time:.2f} seconds") # <<< MODIFIED
    return results

# Execute analysis with state/hub selection
if __name__ == "__main__":
    # Create the results table if it doesn't exist or needs updates
    create_var_results_table(service_name="EntradeDev", table_name="Credit_Exposure_Calc")

    # 1. Find all Load Profile IDs that need calculation
    logging.info("🔍 Identifying Load Profiles that require calculation...") # <<< MODIFIED
    profiles_to_process = get_load_profiles_to_calculate(service_name="EntradeDev", table_name="Credit_Exposure_Calc")

    if not profiles_to_process:
        logging.info("\n✅ All Load Profiles are up to date. No calculations needed.") # <<< MODIFIED
    else:
        # 2. Loop through each profile and process it
        for profile_inputs in profiles_to_process:
            load_profile_id = profile_inputs.get("Load_Profile_ID")
            all_terms_for_profile = profile_inputs.get("terms", [])
            
            logging.info(f"\n🚀 Processing Load Profile ID: {load_profile_id}") # <<< MODIFIED
            logging.info(f"  Input parameters: {profile_inputs}") # <<< MODIFIED

            # 3. For the current profile, check which specific terms need calculation
            logging.info(f"  Checking calculation status for terms: {all_terms_for_profile}") # <<< MODIFIED
            terms_to_calculate = check_calculation_flags_for_profile(
                load_profile_id=load_profile_id,
                all_terms=all_terms_for_profile,
                service_name="EntradeDev", 
                table_name="Credit_Exposure_Calc"
            )

            if not terms_to_calculate:
                logging.info(f"  ✅ All terms for Load Profile {load_profile_id} are already calculated. Skipping.") # <<< MODIFIED
                # Even if no terms needed calculation, we should mark the whole profile as done.
                update_calculation_flag_for_profile(load_profile_id, service_name="EntradeDev", table_name="Credit_Exposure_Calc")
                continue

            logging.info(f"\n  ▶️ Running calculations for terms: {terms_to_calculate}") # <<< MODIFIED
            
            # 4. Run the analysis for the terms that need it
            results = run_multiterm_analysis(
                state=profile_inputs.get("state"),
                terms_to_run=terms_to_calculate,
                start_month_input=profile_inputs.get("start_month_input"),
                Billing_Type=profile_inputs.get("Billing_Type"),
                Monthly_Volume_12Months=profile_inputs.get("Monthly_Volume_12Months")
            )
            
            # 5. Export results to SQL and update flags
            if results:
                export_success = export_results_to_sql(
                    results=results,
                    load_profile_id=load_profile_id,
                    service_name="EntradeDev",
                    table_name="Credit_Exposure_Calc"
                )
                
                if export_success:
                    logging.info(f"\n  ✅ VAR analysis for Load Profile {load_profile_id} completed and exported.") # <<< MODIFIED
                    
                    # 6. Update the calculation flag for the entire Load Profile ID
                    logging.info(f"\n  🏁 Updating calculation flag for Load Profile {load_profile_id}...") # <<< MODIFIED
                    flag_updated = update_calculation_flag_for_profile(
                        load_profile_id,
                        service_name="EntradeDev",
                        table_name="Credit_Exposure_Calc"
                    )
                else:
                    logging.error(f"\n  ❌ VAR analysis for Load Profile {load_profile_id} completed but SQL export failed!") # <<< MODIFIED
            else:
                logging.warning(f"\n  ❌ VAR analysis failed for Load Profile {load_profile_id} - no results to export.") # <<< MODIFIED

        logging.info("\n\n🎉 All pending calculations are complete!") # <<< MODIFIED