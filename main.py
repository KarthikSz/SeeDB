from seedb import SeeDB
import numpy as np
from db_utils import *

def main(db_name):
    user = 'postgres'
    measures = [ 'age' , 'capital_gain' , 'capital_loss' , 'hours_per_week' ]
    dimensions = [ 'workclass' , 'education' , 'occupation' , 'relationship' ,
                'race' , 'native_country' , 'salary' ]
    table_name = 'census'

    seedb = SeeDB(db_name, user, measures, dimensions, table_name)

    # recommend_views test
    ref_dataset = "marital_status in (' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse')"
    query_dataset = "marital_status in (' Divorced', ' Never-married', ' Separated', ' Widowed')"
    views = seedb.recommend_views( query_dataset , ref_dataset , 5 )
    labels = [ 'Married' , 'Unmarried' ]
    seedb.visualize( views , query_dataset , ref_dataset ,  labels ,
    folder_path = 'visualizations/sex/' )

if __name__ == '__main__':
    db_name = 'seedb'
    main(db_name)