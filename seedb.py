from typing import List
from DB_Functions import DB_Functions
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from scipy.stats import entropy
import CTE_fstrings

class SeeDB:
    def __init__( self , db_name , user , measure_attributes , dimension_attributes , table_name ):
        self.db_handler = DB_Functions()
        self.db = self.db_handler.connect_database( db_name , user )
        self.num_rows = self.db_handler.select_query( self.db , ( 'select count(*) from ' + table_name ))[ 0 ][ 0 ]
        self.measure_attributes = measure_attributes
        self.dimension_attributes = dimension_attributes
        self.table_name = table_name
        self.functions = [ 'avg' , 'min' , 'max' , 'sum' , 'count' ]

    def kl_divergence( self , p1 , p2 ):
        eps = 1e-5
        p1 = p1/( np.sum( p1 ) + eps )
        p2 = p2/( np.sum( p2 ) + eps )
        p1[ np.where( p1 < eps ) ] = eps
        p2[ np.where( p2 < eps )] = eps
        kl_divg = entropy( p1 , p2 )
        return kl_divg

    def populate_candidate_params( self ):
        candidate_views = dict()
        for dimension_attribute in self.dimension_attributes:
            for measure in self.measure_attributes:
                for func in self.functions:
                    if dimension_attribute not in candidate_views:
                        candidate_views[ dimension_attribute ] = dict()
                    if measure not in candidate_views[ dimension_attribute ]:
                        candidate_views[ dimension_attribute ][ measure ] = set()
                    candidate_views[ dimension_attribute ][ measure ].add( func )
        return candidate_views

    def get_recommended_views( self , candidate_views ):
        recommended_views = []
        for dimension_attribute in candidate_views:
            for measure in candidate_views[ dimension_attribute ]:
                for func in candidate_views[ dimension_attribute ][ measure ]:
                    recommended_views.append( ( dimension_attribute , measure , func ) )
        return recommended_views

    def get_selections( self , candidate_views , dimension_attribute , query_dataset_cond , reference_dataset_cond ):
        query_selections = ''
        ref_selections = ''
        for measure in candidate_views[ dimension_attribute ]:
            for func in candidate_views[ dimension_attribute ][ measure ]:
                # Create selections for query and reference dataset with unique labels
                query_func = CTE_fstrings.query_func_cte_fstring.format( func = func, 
                    cond = query_dataset_cond , 
                    measure = measure
                )

                ref_func = CTE_fstrings.ref_func_cte_fstring.format(
                    func = func ,
                    cond = reference_dataset_cond ,
                    measure = measure
                )

                query_selections += '{}, '.format( query_func )
                ref_selections += '{}, '.format( ref_func )

                # Remove the trailing comma and space
        all_selections = query_selections + ref_selections[ : -2 ]
        return all_selections

    def delete_pruned_views( self , candidate_views , mappings_distidx_view , pruned_view_indexes , dist_views  ):
        pruned_views = [ mappings_distidx_view[ idx ] for idx in pruned_view_indexes ]
        print( len( dist_views ) , len( pruned_view_indexes ) )
        for dimension_attribute, measure, func in pruned_views:
            candidate_views[ dimension_attribute ][ measure ].remove( func )
            if len( candidate_views[ dimension_attribute ][ measure ] ) == 0:
                del candidate_views[ dimension_attribute ][ measure ]
                if len( candidate_views[ dimension_attribute ] ) == 0:
                    del candidate_views[ dimension_attribute ]
        return( candidate_views )

    def split_combined_df( self , combine_df ):
        r = self.split_dataframe_by_keyword( combine_df , 'ref' )
        r = np.array( r ) 
        q = self.split_dataframe_by_keyword( combine_df , 'query' )
        q = np.array( q )
        return( [ r , q  ] )

    def recommend_views( self , query_dataset_cond , reference_dataset_cond , n_phases = 10 ):
        '''
        Inputs:
        query_dataset_cond     : what goes in the WHERE sql clause
                                 to get query dataset
        reference_dataset_cond : what goes in the WHERE sql clause
                                 to get reference dataset
        '''
        dist_fn_pruning = self.kl_divergence   # distance function for pruning
        candidate_views = self.populate_candidate_params()

        itr_phase = 0
        for start, end in self.phase_idx_generator( n_phases ):
            itr_phase += 1
            itr_view = -1
            dist_views = []
            mappings_distidx_view = dict()

            # Initialize a CTE to ensure all attributes are represented in the output
            for dimension_attribute in candidate_views:
                attributes_cte = CTE_fstrings.attributes_cte_fstring.format( attribute = dimension_attribute ,
                table_name = self.table_name )
                all_selections = self.get_selections( candidate_views , dimension_attribute , query_dataset_cond , reference_dataset_cond )

                # Generate the combined SQL query for the current attribute using the attributes CTE
                combined_query = CTE_fstrings.combined_query_cte_fstring.format(
                    attributes_cte = attributes_cte ,
                    all_selections = all_selections ,
                    table_name = self.table_name ,
                    attribute = dimension_attribute ,
                    start = start ,
                    end = end
                )

                print( "Combined Query:" , combined_query )
                combine_df = self.db_handler.fetch_data( self.db , combined_query , query_params = None )
                [ r , q ] = self.split_combined_df( combine_df )
                ## for pruning
                itr_col = -1
                for measure in candidate_views[ dimension_attribute ]:
                    for func in candidate_views[ dimension_attribute ][ measure ]:
                        itr_view += 1
                        itr_col += 1
                        d = dist_fn_pruning( q[ : , itr_col ] , r[ : , itr_col ] )
                        dist_views.append( d )
                        mappings_distidx_view[ itr_view ] = ( dimension_attribute , measure , func )

            ## prune
            pruned_view_indexes = self.prune( dist_views , itr_phase )

            candidate_views = self.delete_pruned_views( candidate_views , mappings_distidx_view , pruned_view_indexes ,
            dist_views  )

        recommended_views = self.get_recommended_views( candidate_views )
        return recommended_views
    
    def _make_combined_view_query( self , selections , table_name , dimension_attribute , start , end ):
        return CTE_fstrings.make_combined_view_query_fstring.format(
            attribute = dimension_attribute ,
            selections = selections ,
            table_name = table_name ,
            start = start ,
            end = end
        )

    def split_dataframe_by_keyword( self , df , keyword ):
        # Filter columns that contain the keyword and always include the first column (assuming it's the key attribute here)
        filtered_columns =  [ col for col in df.columns if keyword in col ] 
        # Drop duplicates to ensure the first column is not repeated if it contains the keyword
        filtered_columns = list( dict.fromkeys( filtered_columns ) )
        return df[ filtered_columns ]

    def _make_view_query( self , selections , table_name , cond , dimension_attribute , start , end ):
        return ' '.join(['with attrs as (select distinct('
                            + dimension_attribute + ') as __atr__',
                            'from', table_name , ')',
                        'select', selections ,
                        'from', 'attrs left outer join', table_name ,
                            'on', '__atr__ =', dimension_attribute ,
                            'and', cond ,
                            'and id>=' + str( start ) , 'and' , 'id<' + str( end ) ,
                        'group by __atr__' ,
                        'order by __atr__' ] )


    def generate_plot( self , table_query , val_query , val_reference ,  labels , dimension_attribute , function , measure , label_query , folder_path ):
        plt.figure()
        n_groups = table_query.shape[ 0 ]
        index = np.arange( n_groups )
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar( index , val_query , bar_width ,
                            alpha = opacity ,
                            color = 'b',
                            label = labels[ 0 ] )

        rects2 = plt.bar( index + bar_width , val_reference , bar_width ,
                            alpha = opacity ,
                            color = 'g' ,
                            label = labels[ 1 ] )

        plt.xlabel( dimension_attribute )
        print( function + '(' + measure + ')' )
        plt.ylabel( function + '(' + measure + ')' ) 
        plt.xticks( index + bar_width , tuple( label_query ) , rotation = 90 )
        plt.legend()
        plt.tight_layout()
        plt.savefig( folder_path + dimension_attribute + '_' + measure + '_' + function + '.png', dpi = 300 )
        plt.close()

    def visualize( self , views , query_dataset_cond , reference_dataset_cond ,
            labels = None , folder_path = 'visualizations/' ):
        '''

        '''
        if labels == None:
            labels = ['Query','Reference']
        for view in views:
            print(view)
            dimension_attribute, measure, function = view
            # get the query table result
            selection = '__atr__'+' , coalesce('+function + '(' + measure + '), 0) '
            query_dataset_query = self._make_view_query(selection,
                    self.table_name, query_dataset_cond, dimension_attribute, 0, self.num_rows)
            reference_dataset_query = self._make_view_query(selection,
                    self.table_name, reference_dataset_cond, dimension_attribute, 0, self.num_rows)
            table_query = np.array( self.db_handler.select_query(self.db, query_dataset_query))
            table_reference = np.array( self.db_handler.select_query(self.db, reference_dataset_query))
            val_query = table_query[ : , 1 ].astype( float )
            label_query = table_query[ : , 0 ]
            val_reference = table_reference[:,1].astype(float)
            a = val_query
            b = val_reference
            self.generate_plot( table_query , val_query , val_reference ,  labels , dimension_attribute , function , measure ,
            label_query , folder_path )

    def phase_idx_generator(self, n_phases=10):
        '''
        Divides the data into mini-batches (indexes ??)
        and returns the mini-batches (indexes ??) for query.
        With feauture-first grouping ???
        '''
        batch_size = np.floor(self.num_rows/n_phases).astype(int)
        return [ ( batch_size*i , batch_size*(i+1) ) for i in range( n_phases ) ]

    def calc_conf_interval( self , iter_phase, N_phase = 10 ):
        '''
        Divides the data into mini-batches (indexes ??)
        and returns the mini-batches (indexes ??) for query.
        With feauture-first grouping ???
        '''
        delta = 0.05
        a = 1.-(iter_phase/N_phase)
        b = 2*np.log(np.log(iter_phase+1))
        c = np.log(np.pi**2/(3*delta))
        d = 0.5*1/(iter_phase+1)
        conf_error = np.sqrt(a*(b+c)*d)
        return conf_error

    def prune( self, kl_divg, iter_phase, N_phase = 10, delta = 0.05, k = 5 ):
        '''
        input: a list of KL divergences of the candidate views, batch number
        output: a list of indices of the views that should be discarded
        '''
        kl_divg = np.array( kl_divg )
        if iter_phase == 1:
            return []
        N = len( kl_divg )

        # sort the kl divergences
        kl_sorted = np.sort( kl_divg )[ : : -1 ]
        index = np.argsort(kl_divg)[ : : -1 ]
        if iter_phase == N_phase:
            return index[ k : ]

        # Calculate the confidence interval
        conf_error = self.calc_conf_interval( iter_phase, N_phase )

        min_kl_divg = kl_sorted[ k - 1 ] - conf_error
        for i in range( k , N ):
            if kl_sorted[ i ] + conf_error < min_kl_divg:
                return index[ i : ]
        return []

if __name__ == '__main__':
    seedb = SeeDB( 'mini_project' , 'postgres' , [ 'fnlwgt' , 'age' , 'capital_gain' , 'capital_loss' , 'hours_per_week' ] , [ 'workclass' , 'education' , 'occupation' , 
    'relationship' , 'race' , 'native_country' , 'salary' ] , 'census' )
    # recommend_views test
    ref_dataset = "marital_status in (' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse')"
    query_dataset = "marital_status in (' Divorced', ' Never-married', ' Separated', ' Widowed')"
    views = seedb.recommend_views( query_dataset , ref_dataset , 5 )
    labels = [ 'Married' , 'Unmarried' ]
    seedb.visualize( views , query_dataset , ref_dataset ,  [ 'Married' , 'Unmarried' ] ,
    folder_path = 'visualizations/sex/' )

