#!/usr/bin/env python
# import
## batteries
import os
import re
import sys
import logging
## 3rd party
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.sql import text
from pypika import Table, Query

# functions
def BL_create_engine() -> Engine:
    """
    Create db engine for Benchling database.    
    Returns:
        db: SQLAlchemy engine object
    """
    logging.info(f'PY_CONFIG_ACTIVE: {os.environ.get("PY_CONFIG_ACTIVE")}')
    # set test/prod benchling tenant
    tenant = 'TEST' if os.environ.get('PY_CONFIG_ACTIVE') == 'TEST' else 'PROD'
    hostname = 'test' if tenant == 'TEST' else ''

    # check if db username and password are set
    if os.environ.get(f'BENCHLING_{tenant}_DB_USERNAME') is None:
        logging.error(f'No Benchling {tenant} database username found in environment variables')
        sys.exit(1)
    if os.environ.get(f'BENCHLING_{tenant}_DB_PASSWORD') is None:
        logging.error(f'No Benchling {tenant} database password found in environment variables')
        sys.exit(1)

    # create db string
    db_string = "postgresql://{user}:{password}@{host}:{port}/{dbname}"
    db_string = db_string.format(
        host=f'postgres-warehouse.arcinstitute{hostname}.benchling.com',
        dbname='warehouse',
        port='5432',
        user='u$' + os.environ[f'BENCHLING_{tenant}_DB_USERNAME'],
        password=os.environ[f'BENCHLING_{tenant}_DB_PASSWORD']
    )
    # create db connection engine
    return create_engine(db_string)

def get_benchling_entities(conn: Connection, plate_name: str = None) -> pd.DataFrame:
    # Define the table
    organoid = Table('organoidsphereiod_lot$raw')
    plate = Table('plate$raw')
    well = Table('well$raw')

    # plate => well => organoid
    query = Query \
        .from_(plate) \
        .where(plate["name"] == plate_name) \
        .join(well) \
        .on(plate["id"] == well["plate_id$"]) \
        .join(organoid) \
        .on(well["well_entity"] == organoid["id"]) \
        .select(
            well["name$"].as_("well_name"), 
            organoid["name$"].as_("organoid_name"),
            organoid["days_in_culture"],
            organoid["reprogrammingdifferentiation_method"],
            organoid["extracellular_matrix"],
            organoid["media_composition"],
            organoid["escipsc_line"],
            organoid["growth_factorssmall_molecules"],
            organoid["plasmid_lot"],
            organoid["grna_lot"],
            organoid["viral_vector_lot"],
            organoid["frozen_down"],
            organoid["date_frozen_down"],
            organoid["quantity"]
        )

    # Execute the query
    res = conn.execute(text(query.get_sql()))

    # Fetch the results and convert them to a DataFrame
    DF = pd.DataFrame(res.fetchall(), columns=res.keys())

    # Format
    ## Split well_name into plate and well (separator is ":")
    DF[['plate_name', 'well_name']] = DF['well_name'].str.split(':', expand=True)
    ## Add a zero if well_name is a letter and single digit
    regex = re.compile(r'^([A-Z])([0-9])$')
    DF['well_name'] = DF['well_name'].apply(lambda x: regex.sub("\g<1>0\g<2>", x))

    # Return the DataFrame
    return DF


def main():
    # create db engine
    db = BL_create_engine()
    # create db connection
    conn = db.connect()
    # query db
    DF = get_benchling_entities(conn, "Sph08302024_Lot_Test_Plate1")
    # remove connection
    conn.close()
    

if __name__ == '__main__':
    load_dotenv()
    main()