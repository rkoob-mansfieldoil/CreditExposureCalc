import requests
import json
from typing import List, Dict, Any
import pyodbc
import keyring
import re

# =========================================================
# CREDENTIALS CONFIGURATION - MODIFY AS NEEDED
# =========================================================
# HubSpot API Bearer Token
HUBSPOT_BEARER_TOKEN = "XXX"  # Replace with your actual token

# SQL Server Connection Parameters
SQL_SERVER = "entradev6db.database.windows.net"  # SQL Server address
SQL_DATABASE = "MPGAPIDB"                        # Database name
SQL_CREDENTIAL_NAME = "MPGAPIDB"                 # Windows Credential Manager name
SQL_TABLE_NAME = "Credit_Exposure_Calc"      # Replace with your actual table name
# =========================================================

# Enable debugging
DEBUG = False

class HubspotAPI:
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.base_url = "https://api.hubapi.com"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
    
    def get_all_custom_objects(self, object_type: str, properties: List[str], 
                              filter_property: str = None, filter_value: str = None) -> List[Dict[str, Any]]:
        """
        Get all custom objects of a specific type with optional filtering
        """
        url = f"{self.base_url}/crm/v3/objects/{object_type}"
        params = {
            "properties": ",".join(properties),
            "limit": 100
        }
        
        all_objects = []
        has_more = True
        after = None
        
        while has_more:
            if after:
                params["after"] = after
                
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                if DEBUG:
                    print(f"Error getting objects: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            
            # Filter results if filter is provided
            if filter_property and filter_value:
                filtered_results = [
                    obj for obj in data.get("results", [])
                    if obj.get("properties", {}).get(filter_property) == filter_value
                ]
                all_objects.extend(filtered_results)
            else:
                all_objects.extend(data.get("results", []))
                
            has_more = data.get("paging", {}).get("next", {}).get("after") is not None
            after = data.get("paging", {}).get("next", {}).get("after")
            
        if DEBUG:
            print(f"Found {len(all_objects)} objects")
        return all_objects
    
    def get_associated_objects(self, object_type: str, object_id: str, 
                              to_object_type: str) -> List[str]:
        """
        Get IDs of objects associated with the given object
        """
        url = f"{self.base_url}/crm/v4/objects/{object_type}/{object_id}/associations/{to_object_type}"
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            if DEBUG:
                print(f"Error getting associations: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        associated_ids = [assoc.get("toObjectId") for assoc in data.get("results", [])]
        
        if DEBUG and associated_ids:
            print(f"Found {len(associated_ids)} associated deal IDs for object {object_id}: {associated_ids}")
            
        return associated_ids
    
    def get_objects_by_ids(self, object_type: str, object_ids: List[str], 
                          properties: List[str]) -> List[Dict[str, Any]]:
        """
        Get objects by their IDs with specific properties
        """
        if not object_ids:
            return []
            
        url = f"{self.base_url}/crm/v3/objects/{object_type}/batch/read"
        
        # Process in batches of 100 (HubSpot's limit)
        all_objects = []
        
        for i in range(0, len(object_ids), 100):
            batch_ids = object_ids[i:i+100]
            
            payload = {
                "properties": properties,
                "inputs": [{"id": obj_id} for obj_id in batch_ids]
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code != 200:
                if DEBUG:
                    print(f"Error getting objects by IDs: {response.status_code} - {response.text}")
                continue
                
            data = response.json()
            all_objects.extend(data.get("results", []))
            
        if DEBUG:
            print(f"Retrieved {len(all_objects)} objects by ID")
            for obj in all_objects:
                print(f"  Deal ID: {obj.get('id')}, Name: {obj.get('properties', {}).get('dealname', 'N/A')}")
        return all_objects

class SQLIntegration:
    def __init__(self, server, database, credential_name, table_name):
        self.server = server
        self.database = database
        self.credential_name = credential_name
        self.table_name = table_name
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to SQL Server using credentials from Windows Credential Manager"""
        try:
            # Get credentials from Windows Credential Manager
            credential = keyring.get_credential(self.credential_name, None)
            
            if credential is None:
                print(f"Error: No credentials found for {self.credential_name} in Windows Credential Manager")
                return False
                
            username = credential.username
            password = credential.password
            
            # Connect to SQL Server using SQL Server Authentication
            connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={username};PWD={password}'
            self.conn = pyodbc.connect(connection_string)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"Error connecting to SQL Server: {e}")
            return False
            
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def check_existing_record(self, load_profile_id, term_split):
        """
        Check if a record with the given Load_Profile_ID AND Term_Split exists 
        and check its Hubspot_Patch_Flag.
        
        Returns True if we should insert the record:
        - No record exists with that Load_Profile_ID and Term_Split, OR
        - A record exists but its Hubspot_Patch_Flag is NOT NULL
        
        Returns False if we should not insert:
        - A record exists with that Load_Profile_ID and Term_Split, and its Hubspot_Patch_Flag is NULL
        """
        if not self.cursor:
            return True
            
        # Handle NULL term_split in the query
        if term_split is None:
            query = f"""
            SELECT Hubspot_Patch_Flag 
            FROM {self.table_name} 
            WHERE Load_Profile_ID = ? AND Term_Split IS NULL
            """
            params = (load_profile_id,)
        else:
            query = f"""
            SELECT Hubspot_Patch_Flag 
            FROM {self.table_name} 
            WHERE Load_Profile_ID = ? AND Term_Split = ?
            """
            params = (load_profile_id, term_split)
        
        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchone()
            
            # If no record exists, return True (we should insert)
            if result is None:
                return True
                
            # If Hubspot_Patch_Flag is not NULL, return True (we should insert)
            if result[0] is not None:
                return True
                
            # Otherwise, return False (we should not insert)
            return False
        except Exception as e:
            print(f"Error checking existing record: {e}")
            # If there's an error, assume we should insert
            return True
            
    def insert_exposure_data(self, data_rows):
        """Insert exposure data into SQL table"""
        if not self.cursor:
            return False
            
        query = f"""
        INSERT INTO {self.table_name} (
            Deal_ID, Deal_Name, Contract_Start_Date, Billing_Type, Term_Options, 
            Term_Split, Load_Profile_ID, Load_Profile_Name, Utility_State, 
            Exposure_calculation_status, Jan_Dth, Feb_Dth, Mar_Dth, Apr_Dth, May_Dth, 
            Jun_Dth, Jul_Dth, Aug_Dth, Sep_Dth, Oct_Dth, Nov_Dth, Dec_Dth, Total_Load_Shape_Dth,
            Python_Calc_Flag, PFE, MTM, Fixed_Price_AR, Variable_Price_AR, Hubspot_Patch_Flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            rows_inserted = 0
            rows_skipped = 0
            
            for row in data_rows:
                # Get Load_Profile_ID from the row (index 6 based on the INSERT statement)
                load_profile_id = row[6]
                # Get Term_Split from the row (index 5 based on the INSERT statement)
                term_split = row[5]
                
                # Debug information about the row being inserted
                if DEBUG:
                    print(f"Attempting to insert row - Deal_ID: {row[0]}, Load_Profile_ID: {load_profile_id}, Term_Split: {term_split}")
                
                # Check if we should insert this record based on both Load_Profile_ID and Term_Split
                if self.check_existing_record(load_profile_id, term_split):
                    # Convert deal_id to an integer if it's a string and not None
                    # This is important if the SQL column is an integer type
                    if row[0] is not None and isinstance(row[0], str):
                        try:
                            row = list(row)  # Convert tuple to list so we can modify it
                            row[0] = int(row[0])
                        except ValueError:
                            # If conversion fails, keep the original value
                            pass
                    
                    self.cursor.execute(query, row)
                    rows_inserted += 1
                    if DEBUG:
                        print(f"Inserted row with Deal_ID: {row[0]}, Load_Profile_ID: {load_profile_id}, Term_Split: {term_split}")
                else:
                    rows_skipped += 1
                    if DEBUG:
                        print(f"Skipped row with Load_Profile_ID: {load_profile_id}, Term_Split: {term_split} (record exists with NULL Hubspot_Patch_Flag)")
            
            self.conn.commit()
            print(f"Rows inserted: {rows_inserted}, Rows skipped: {rows_skipped}")
            return True
        except Exception as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            return False

def prepare_sql_data(exposure_object, associated_deals):
    """
    Prepare data for SQL insertion based on mapping logic
    """
    sql_rows = []
    
    # Extract properties from the exposure object
    obj_id = exposure_object.get("id")
    obj_properties = exposure_object.get("properties", {})
    
    # Map exposure object properties
    load_profile_id = obj_properties.get("hs_object_id", "")
    load_profile_name = obj_properties.get("leusl_name", "")
    utility_state = obj_properties.get("utility_state", "")
    exposure_calculation_status = obj_properties.get("exposure_calculation_status", "")
    
    # Get Billing_Type from exposure object (moved from deal level to exposure level)
    billing_type = obj_properties.get("type_of_billing", "")
    
    # Map DTH values, converting to integers when possible
    jan_dth = obj_properties.get("jan__dth_", None)
    feb_dth = obj_properties.get("feb__dth_", None)
    mar_dth = obj_properties.get("mar__dth_", None)
    apr_dth = obj_properties.get("apr__dth_", None)
    may_dth = obj_properties.get("may__dth_", None)
    jun_dth = obj_properties.get("jun__dth_", None)
    jul_dth = obj_properties.get("jul__dth_", None)
    aug_dth = obj_properties.get("aug__dth_", None)
    sep_dth = obj_properties.get("sep__dth_", None)
    oct_dth = obj_properties.get("oct__dth_", None)
    nov_dth = obj_properties.get("nov__dth_", None)
    dec_dth = obj_properties.get("dec__dth_", None)
    total_load_shape_dth = obj_properties.get("total_load_shape__dth_", None)
    
    # Convert DTH values to integers when possible
    dth_values = [jan_dth, feb_dth, mar_dth, apr_dth, may_dth, jun_dth, 
                  jul_dth, aug_dth, sep_dth, oct_dth, nov_dth, dec_dth, total_load_shape_dth]
    
    for i, val in enumerate(dth_values):
        try:
            if val is not None and isinstance(val, str) and val.strip():
                dth_values[i] = int(float(val))
            else:
                dth_values[i] = None
        except (ValueError, TypeError):
            dth_values[i] = None
    
    # Unpack DTH values
    jan_dth, feb_dth, mar_dth, apr_dth, may_dth, jun_dth, jul_dth, aug_dth, sep_dth, oct_dth, nov_dth, dec_dth, total_load_shape_dth = dth_values
    
    # Debug information about the exposure object
    if DEBUG:
        print(f"Processing exposure object ID: {obj_id}, Load Profile ID: {load_profile_id}")
        print(f"Billing Type: {billing_type}")
        print(f"Associated deals count: {len(associated_deals)}")
        for deal in associated_deals:
            print(f"  Associated deal: {deal.get('id')}, Name: {deal.get('properties', {}).get('dealname', 'N/A')}")
    
    # If no associated deals, add a row with deal fields blank
    if not associated_deals:
        if DEBUG:
            print(f"No associated deals for exposure object ID: {obj_id}")
            
        row_base = [
            None,                   # Deal_ID
            None,                   # Deal_Name
            None,                   # Contract_Start_Date
            billing_type,           # Billing_Type (now from exposure level)
            None,                   # Term_Options
            None,                   # Term_Split
            load_profile_id,        # Load_Profile_ID
            load_profile_name,      # Load_Profile_Name
            utility_state,          # Utility_State
            exposure_calculation_status,  # Exposure_calculation_status
            jan_dth, feb_dth, mar_dth, apr_dth, may_dth,
            jun_dth, jul_dth, aug_dth, sep_dth, oct_dth, nov_dth, dec_dth, total_load_shape_dth, 
            None,                   # Python_Calc_Flag
            None,                   # PFE
            None,                   # MTM
            None,                   # Fixed_Price_AR
            None,                   # Variable_Price_AR
            None                    # Hubspot_Patch_Flag
        ]
        sql_rows.append(row_base)
        return sql_rows
    
    # Process each associated deal
    for deal in associated_deals:
        deal_id = deal.get("id")
        deal_properties = deal.get("properties", {})
        
        # Debug information about the deal
        if DEBUG:
            print(f"Processing deal ID: {deal_id} for exposure object ID: {obj_id}")
        
        deal_name = deal_properties.get("dealname", "")
        contract_start_date = deal_properties.get("flow_start_month", "")
        # Billing_type is now from exposure level, not deal level
        term_options = deal_properties.get("term_options", "")
        
        # Handle Term_Split logic - create a row for each term in term_options
        terms = []
        if term_options and isinstance(term_options, str) and term_options.strip():
            # Split by comma and remove any whitespace
            terms = [term.strip() for term in term_options.split(",") if term.strip()]
        
        if not terms:
            # If no terms, add a single row with term_split as null
            if DEBUG:
                print(f"No terms found for deal ID: {deal_id}, adding single row")
                
            row_base = [
                deal_id,               # Deal_ID
                deal_name,             # Deal_Name
                contract_start_date,   # Contract_Start_Date
                billing_type,          # Billing_Type (now from exposure level)
                term_options,          # Term_Options
                None,                  # Term_Split
                load_profile_id,       # Load_Profile_ID
                load_profile_name,     # Load_Profile_Name
                utility_state,         # Utility_State
                exposure_calculation_status,  # Exposure_calculation_status
                jan_dth, feb_dth, mar_dth, apr_dth, may_dth,
                jun_dth, jul_dth, aug_dth, sep_dth, oct_dth, nov_dth, dec_dth, total_load_shape_dth, 
                None,                  # Python_Calc_Flag
                None,                  # PFE
                None,                  # MTM
                None,                  # Fixed_Price_AR
                None,                  # Variable_Price_AR
                None                   # Hubspot_Patch_Flag
            ]
            sql_rows.append(row_base)
        else:
            # Create a row for each term
            if DEBUG:
                print(f"Found {len(terms)} terms for deal ID: {deal_id}")
                
            for term in terms:
                try:
                    term_split = int(term)
                except ValueError:
                    term_split = None
                
                if DEBUG:
                    print(f"Adding row with deal ID: {deal_id}, term: {term}")
                    
                row = [
                    deal_id,               # Deal_ID
                    deal_name,             # Deal_Name
                    contract_start_date,   # Contract_Start_Date
                    billing_type,          # Billing_Type (now from exposure level)
                    term_options,          # Term_Options
                    term_split,            # Term_Split
                    load_profile_id,       # Load_Profile_ID
                    load_profile_name,     # Load_Profile_Name
                    utility_state,         # Utility_State
                    exposure_calculation_status,  # Exposure_calculation_status
                    jan_dth, feb_dth, mar_dth, apr_dth, may_dth,
                    jun_dth, jul_dth, aug_dth, sep_dth, oct_dth, nov_dth, dec_dth, total_load_shape_dth, 
                    None,                  # Python_Calc_Flag
                    None,                  # PFE
                    None,                  # MTM
                    None,                  # Fixed_Price_AR
                    None,                  # Variable_Price_AR
                    None                   # Hubspot_Patch_Flag
                ]
                sql_rows.append(row)
    
    return sql_rows

def main():
    # Use hard-coded bearer token
    api = HubspotAPI(HUBSPOT_BEARER_TOKEN)
    
    # Define properties to fetch
    custom_object_properties = [
        "hs_object_id", "leusl_name", "utility_state", "exposure_calculation_status",
        "jan__dth_", "feb__dth_", "mar__dth_", "apr__dth_", "may__dth_", 
        "jun__dth_", "jul__dth_", "aug__dth_", "sep__dth_", 
        "oct__dth_", "nov__dth_", "dec__dth_", "total_load_shape__dth_",
        "type_of_billing"  # Added type_of_billing to exposure object properties
    ]
    
    deal_properties = [
        "hs_object_id", "dealname", "flow_start_month", "term_options"
        # Removed "type_of_billing" from deal properties since it's now at exposure level
    ]
    
    # Get all custom objects with "Running Exposure" status
    custom_objects = api.get_all_custom_objects(
        "2-47241908", 
        custom_object_properties,
        "exposure_calculation_status", 
        "Running Exposure"
    )
    
    if not custom_objects:
        print("No custom objects found with 'Running Exposure' status")
        return
    
    # Process each exposure object and its deals
    sql_data = []
    
    for obj in custom_objects:
        obj_id = obj.get("id")
        
        # Get associated deals for this object
        associated_deal_ids = api.get_associated_objects(
            "2-47241908", 
            obj_id, 
            "deal"
        )
        
        if not associated_deal_ids:
            # If no associated deals, still process the exposure object
            obj_data = prepare_sql_data(obj, [])
            sql_data.extend(obj_data)
            continue
        
        # Get deal details for this object's associated deals
        associated_deals = api.get_objects_by_ids(
            "deal", 
            associated_deal_ids, 
            deal_properties
        )
        
        # Prepare data for this object and its deals
        obj_data = prepare_sql_data(obj, associated_deals)
        sql_data.extend(obj_data)
    
    if DEBUG:
        print(f"Prepared {len(sql_data)} rows for SQL insertion")
        # Print a few sample rows to verify deal IDs are present
        for i, row in enumerate(sql_data[:5]):
            print(f"Sample row {i+1}: Deal_ID={row[0]}, Load_Profile_ID={row[6]}, Term_Split={row[5]}, Billing_Type={row[3]}")
    
    # Initialize SQL integration with credentials from Windows Credential Manager
    sql = SQLIntegration(SQL_SERVER, SQL_DATABASE, SQL_CREDENTIAL_NAME, SQL_TABLE_NAME)
    
    # Connect to SQL Server
    if sql.connect():
        # Insert data
        success = sql.insert_exposure_data(sql_data)
        
        # Close connection
        sql.close()

if __name__ == "__main__":
    main()