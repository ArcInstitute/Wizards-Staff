import pytest
import os
from dotenv import load_dotenv
from wizards_staff.gcp import download_gcp_dir

@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """
    Fixture to download test data from GCP bucket before any tests are run.
    This fixture has session scope, meaning it will run once per test session.
    """
    # load environment variables
    load_dotenv()  
    
    # Set up your bucket information (you might want to use environment variables or hard code them)
    bucket_name = 'arc-genomics-test-data'
    run_name = 'Calcium_AAV-GCAMP_6wk_20240416'
    prefix = os.path.join('wizards-staff', run_name)

    # Local direct
    local_dir = os.path.join('tests', 'data', run_name)

    # Download the dataset, if needed
    if not os.path.exists(local_dir):
        local_dir = download_gcp_dir(bucket_name, prefix)
    
    # Return the directory where the data is downloaded
    return local_dir