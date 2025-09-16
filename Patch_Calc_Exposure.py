import requests
import json
from typing import List, Dict, Any
import pyodbc
import keyring
import re

# =========================================================
# CREDENTIALS CONFIGURATION - MODIFY AS NEEDED
# =========================================================
# HubSpot API Bearer Token - now retrieved from Windows Credential Manager
HUBSPOT_CREDENTIAL_NAME = "HubSpot Sandbox"  # Windows Credential Manager name for HubSpot token

# SQL Server Connection Parameters
SQL_SERVER = "entradev6db.database.windows.net"  # SQL Server address
SQL_DATABASE = "MPGAPIDB"                        # Database name
SQL_CREDENTIAL_NAME = "MPGAPIDB"                 # Windows Credential Manager name
SQL_TABLE_NAME = "Credit_Exposure_Calc"          # Replace with your actual table name

# HubSpot Custom Object Configuration
HUBSPOT_CUSTOM_OBJECT_ID = "2-47241908"         # MODIFY FOR PROD - Custom object type ID
# =========================================================

# Enable debugging
DEBUG = True

def get_hubspot_bearer_token():
    """Get HubSpot bearer token from Windows Credential Manager"""
    try:
        credential = keyring.get_credential(HUBSPOT_CREDENTIAL_NAME, None)
        
        if credential is None:
            print(f"Error: No credentials found for '{HUBSPOT_CREDENTIAL_NAME}' in Windows Credential Manager")
            return None
            
        # The bearer token should be stored as the password
        bearer_token = credential.password
        
        if not bearer_token:
            print(f"Error: No password (bearer token) found for '{HUBSPOT_CREDENTIAL_NAME}' in Windows Credential Manager")
            return None
            
        return bearer_token
    except Exception as e:
        print(f"Error retrieving HubSpot bearer token from Windows Credential Manager: {e}")
        return None

class HubspotAPI:
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.base_url = "https://api.hubapi.com"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
    
    def update_custom_object(self, object_id: str, properties: Dict[str, str]) -> bool:
        """
        Update a custom object with the given properties using PATCH
        """
        url = f"{self.base_url}/crm/v3/objects/{HUBSPOT_CUSTOM_OBJECT_ID}/{object_id}"
        
        payload = {
            "properties": properties
        }
        
        try:
            response = requests.patch(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                if DEBUG:
                    print(f"Successfully updated HubSpot object {object_id}")
                return True
            else:
                if DEBUG:
                    print(f"Error updating HubSpot object {object_id}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            if DEBUG:
                print(f"Exception updating HubSpot object {object_id}: {e}")
            return False

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
    
    def get_load_profiles_to_patch(self) -> List[str]:
        """
        Get all unique Load_Profile_IDs that have records with Hubspot_Patch_Flag NULL or 0
        """
        if not self.cursor:
            return []
            
        query = f"""
        SELECT DISTINCT Load_Profile_ID 
        FROM {self.table_name} 
        WHERE (Hubspot_Patch_Flag IS NULL OR Hubspot_Patch_Flag = 0)
        """
        
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            load_profile_ids = [row[0] for row in results if row[0] is not None]
            
            if DEBUG:
                print(f"Found {len(load_profile_ids)} Load_Profile_IDs to process: {load_profile_ids}")
                
            return load_profile_ids
        except Exception as e:
            print(f"Error getting load profiles to patch: {e}")
            return []
    
    def get_calculation_data_for_load_profile(self, load_profile_id: str) -> List[Dict[str, Any]]:
        """
        Get all calculation data for a specific Load_Profile_ID where 
        Hubspot_Patch_Flag is NULL and Python_Calc_Flag = 1
        """
        if not self.cursor:
            return []
            
        query = f"""
        SELECT Term_Split, PFE, MTM, Fixed_Price_AR, Variable_Price_AR
        FROM {self.table_name}
        WHERE Load_Profile_ID = ? 
        AND (Hubspot_Patch_Flag IS NULL OR Hubspot_Patch_Flag = 0)
        AND Python_Calc_Flag = 1
        ORDER BY Term_Split
        """
        
        try:
            self.cursor.execute(query, (load_profile_id,))
            results = self.cursor.fetchall()
            
            calculation_data = []
            for row in results:
                calculation_data.append({
                    'Term_Split': row[0],
                    'PFE': row[1], 
                    'MTM': row[2],
                    'Fixed_Price_AR': row[3],
                    'Variable_Price_AR': row[4]
                })
            
            if DEBUG:
                print(f"Found {len(calculation_data)} calculation records for Load_Profile_ID {load_profile_id}")
                
            return calculation_data
        except Exception as e:
            print(f"Error getting calculation data for Load_Profile_ID {load_profile_id}: {e}")
            return []
    
    def update_hubspot_patch_flag(self, load_profile_id: str, flag_value: int) -> bool:
        """
        Update Hubspot_Patch_Flag for all records with the given Load_Profile_ID
        where Python_Calc_Flag = 1 and Hubspot_Patch_Flag is NULL or 0
        """
        if not self.cursor:
            return False
            
        query = f"""
        UPDATE {self.table_name} 
        SET Hubspot_Patch_Flag = ?
        WHERE Load_Profile_ID = ? 
        AND Python_Calc_Flag = 1 
        AND (Hubspot_Patch_Flag IS NULL OR Hubspot_Patch_Flag = 0)
        """
        
        try:
            self.cursor.execute(query, (flag_value, load_profile_id))
            rows_updated = self.cursor.rowcount
            self.conn.commit()
            
            if DEBUG:
                print(f"Updated Hubspot_Patch_Flag to {flag_value} for {rows_updated} records with Load_Profile_ID {load_profile_id}")
                
            return True
        except Exception as e:
            print(f"Error updating Hubspot_Patch_Flag for Load_Profile_ID {load_profile_id}: {e}")
            self.conn.rollback()
            return False

def format_currency(value) -> str:
    """
    Format a numeric value as currency with thousand separators
    """
    if value is None:
        return "$0.00"
    
    try:
        # Convert to float first to handle string inputs
        num_value = float(value)
        # Format with commas and 2 decimal places
        return f"${num_value:,.2f}"
    except (ValueError, TypeError):
        return "$0.00"

def create_concatenated_string(calculation_data: List[Dict[str, Any]]) -> str:
    """
    Create the concatenated string from calculation data
    Format: "Term: X PFE: $X,XXX.XX MTM: $X,XXX.XX Fixed Price AR: $X,XXX.XX Variable Price AR: $X,XXX.XX"
    Each row separated by line breaks
    """
    lines = []
    
    for data in calculation_data:
        term = data.get('Term_Split', 'N/A')
        pfe = format_currency(data.get('PFE'))
        mtm = format_currency(data.get('MTM'))
        fixed_price_ar = format_currency(data.get('Fixed_Price_AR'))
        variable_price_ar = format_currency(data.get('Variable_Price_AR'))
        
        line = f"Term: {term} PFE: {pfe} MTM: {mtm} Fixed Price AR: {fixed_price_ar} Variable Price AR: {variable_price_ar}"
        lines.append(line)
    
    # Using \n for line breaks - can be changed to \r\n if needed
    return '\n'.join(lines)

def main():
    # Get HubSpot bearer token from Windows Credential Manager
    bearer_token = get_hubspot_bearer_token()
    if not bearer_token:
        print("Failed to retrieve HubSpot bearer token. Exiting.")
        return
    
    # Initialize API and SQL connections
    api = HubspotAPI(bearer_token)
    sql = SQLIntegration(SQL_SERVER, SQL_DATABASE, SQL_CREDENTIAL_NAME, SQL_TABLE_NAME)
    
    # Connect to SQL Server
    if not sql.connect():
        print("Failed to connect to SQL Server. Exiting.")
        return
    
    try:
        # Get all Load_Profile_IDs that need to be patched
        load_profile_ids = sql.get_load_profiles_to_patch()
        
        if not load_profile_ids:
            print("No Load_Profile_IDs found that need patching.")
            return
        
        print(f"Processing {len(load_profile_ids)} Load_Profile_IDs...")
        
        successful_updates = 0
        failed_updates = 0
        
        for load_profile_id in load_profile_ids:
            if DEBUG:
                print(f"\n--- Processing Load_Profile_ID: {load_profile_id} ---")
            
            # Get calculation data for this Load_Profile_ID
            calculation_data = sql.get_calculation_data_for_load_profile(load_profile_id)
            
            if not calculation_data:
                if DEBUG:
                    print(f"No calculation data found for Load_Profile_ID {load_profile_id}")
                continue
            
            # Create the concatenated string
            exposure_results = create_concatenated_string(calculation_data)
            
            if DEBUG:
                print(f"Created exposure results string for {load_profile_id}:")
                print(f"Length: {len(exposure_results)} characters")
                print("Preview:")
                print(exposure_results[:200] + "..." if len(exposure_results) > 200 else exposure_results)
            
            # Prepare HubSpot update properties
            hubspot_properties = {
                "exposure_calculation___parsed_results": exposure_results,
                "exposure_calculation_status": "Exposure Calculation Complete"
            }
            
            # Update HubSpot custom object
            success = api.update_custom_object(load_profile_id, hubspot_properties)
            
            if success:
                # Update SQL flag to 1 (success)
                sql.update_hubspot_patch_flag(load_profile_id, 1)
                successful_updates += 1
                print(f"✓ Successfully updated Load_Profile_ID: {load_profile_id}")
            else:
                # Update SQL flag to 0 (error - needs review)
                sql.update_hubspot_patch_flag(load_profile_id, 0)
                failed_updates += 1
                print(f"✗ Failed to update Load_Profile_ID: {load_profile_id}")
        
        print(f"\n--- SUMMARY ---")
        print(f"Total processed: {len(load_profile_ids)}")
        print(f"Successful updates: {successful_updates}")
        print(f"Failed updates: {failed_updates}")
        
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")
    finally:
        # Close SQL connection
        sql.close()

if __name__ == "__main__":
    main()