import os
import tempfile
from dotenv import load_dotenv
from google.cloud import storage

def download_gcp_dir(bucket_name: str, prefix: str, outdir: str=None):
    """
    Download all files in a "directory" in a Google Cloud Storage bucket to a local directory.
    Args:
        bucket_name: The name of the bucket.
        prefix: The prefix of the "directory" in the bucket.
        outdir: The local directory to download the files to. If None, a temporary directory is created.  
    Returns:
        The local directory where the files were downloaded
    """
    # Create a temporary directory if outdir is not specified
    if outdir is None:
        outdir = tempfile.mkdtemp()

    # Initialize GCP client
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # List all blobs in the bucket with the specified prefix (simulating a directory)
    blobs = bucket.list_blobs(prefix=prefix)

    # Download each file in the "directory"
    for blob in blobs:
        # Skip directory blobs
        if blob.name.endswith('/') and blob.size == 0:
            continue

        # Remove the prefix from the blob name for local_path
        relative_path = os.path.relpath(blob.name, prefix)
        local_path = os.path.join(outdir, relative_path)

        # Make sure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the blob to the local file
        blob.download_to_filename(local_path)
    
    # Return the output directory
    return outdir

# Example usage
def main():
    output_dir = download_gcp_dir(
        'arc-genomics-test-data', 
        os.path.join('wizards-staff', 'Calcium_AAV-GCAMP_6wk_20240416'),
        outdir = os.path.join('tests', 'data', 'Calcium_AAV-GCAMP_6wk_20240416')
    )

if __name__ == '__main__':
    load_dotenv()
    main()
