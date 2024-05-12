from DB_Functions import DB_Functions
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import CTE_fstrings
from scipy.special import kl_div
import os

class SeeDB:
    def __init__( self , db_name , user , measure_attributes , dimension_attributes , table ):
        self.db_handler = DB_Functions()
        self.database_connection = self.db_handler.connect_database( db_name , user )
        self.num_rows = self.db_handler.ffetch_data( self.database_connection ,
        ( 'select count(*) from ' + table ))[ 0 ][ 0 ]
        self.measure_attributes = measure_attributes
        self.dimension_attributes = dimension_attributes
        self.table = table
        self.aggregate_fns = [ 'max' , 'sum' , 'count' ,  'avg' , 'min' ]

    def compute_kl_div( self , prob_dist_1 , prob_dist_2 ):
        epsilon = 1e-5
        prob_dist_1 = prob_dist_1/( np.sum( prob_dist_1 ) + epsilon )
        prob_dist_2 = prob_dist_2/( np.sum( prob_dist_2 ) + epsilon )
        prob_dist_1 = np.clip( prob_dist_1 , epsilon , None )
        prob_dist_2 = np.clip( prob_dist_2 , epsilon , None )
        kl_divg = np.sum( kl_div( prob_dist_1 , prob_dist_2 ) )
        return kl_divg

    def populate_potential_view_params( self ):
        potential_views = dict()
        for dimension_attribute in self.dimension_attributes:
            for measure_attribute in self.measure_attributes:
                for aggregate_fn in self.aggregate_fns:
                    if dimension_attribute not in potential_views:
                        potential_views[ dimension_attribute ] = dict()
                    if measure_attribute not in potential_views[ dimension_attribute ]:
                        potential_views[ dimension_attribute ][ measure_attribute ] = set()
                    potential_views[ dimension_attribute ][ measure_attribute ].add( aggregate_fn )
        return potential_views

    def get_suggested_views( self , potential_views ):
        suggested_views = []
        for dimension_attribute in potential_views:
            for measure_attribute in potential_views[ dimension_attribute ]:
                for aggregate_fn in potential_views[ dimension_attribute ][ measure_attribute ]:
                    suggested_views.append( ( dimension_attribute , measure_attribute , aggregate_fn ) )
        return suggested_views

    def get_combined_selection_query( self , potential_views , dimension_attribute , user_dataset_condition , reference_dataset_condition ):
        user_selection = ''
        reference_selection = ''
        for measure_attribute in potential_views[ dimension_attribute ]:
            for aggregate_fn in potential_views[ dimension_attribute ][ measure_attribute ]:
                user_fn = CTE_fstrings.user_fn_cte_fstring.format( fn = aggregate_fn , 
                    condition = user_dataset_condition , 
                    measure_attribute = measure_attribute
                )

                reference_fn = CTE_fstrings.reference_fn_cte_fstring.format(
                    fn = aggregate_fn ,
                    condition = reference_dataset_condition ,
                    measure_attribute = measure_attribute
                )

                user_selection += '{}, '.format( user_fn )
                reference_selection += '{}, '.format( reference_fn )

                # Remove the trailing comma and space
        combined_selection = user_selection + reference_selection[ : -2 ]
        return combined_selection

    def delete_pruned_views( self , potential_views , distance_index_mappings_view , pruned_view_indexes , view_distances  ):
        pruned_views = [ distance_index_mappings_view[ id ] for id in pruned_view_indexes ]
        print( len( view_distances ) , len( pruned_view_indexes ) )
        for dimension_attribute, measure_attribute, aggregate_fn in pruned_views:
            potential_views[ dimension_attribute ][ measure_attribute ].remove( aggregate_fn )
            if len( potential_views[ dimension_attribute ][ measure_attribute ] ) == 0:
                del potential_views[ dimension_attribute ][ measure_attribute ]
                if len( potential_views[ dimension_attribute ] ) == 0:
                    del potential_views[ dimension_attribute ]
        return( potential_views )

    def split_combined_df( self , combine_df ):
        r = self.filter_df_keyword( combine_df , 'reference' )
        r = np.array( r ) 
        q = self.filter_df_keyword( combine_df , 'user' )
        q = np.array( q )
        return( [ r , q  ] )

    def recommend_views( self , user_dataset_condition , reference_dataset_condition , num_phases = 10 ):
        '''
        Inputs:
        user_dataset_condition     : what goes in the WHERE sql clause
                                 to get user dataset
        reference_dataset_condition : what goes in the WHERE sql clause
                                 to get reference dataset
        '''
        dist_fn_pruning = self.compute_kl_div   # distance function for pruning
        potential_views = self.populate_potential_view_params()

        itr_phase = 0
        for start, end in self.generate_indices_phase( num_phases ):
            itr_phase += 1
            itr_view = -1
            view_distances = []
            distance_index_mappings_view = dict()

            # Initialize a CTE to ensure all attributes are represented in the output
            for dimension_attribute in potential_views:
                attributes_cte = CTE_fstrings.attributes_cte_fstring.format( attribute = dimension_attribute ,
                table = self.table )
                all_selections = self.get_combined_selection_query( potential_views , dimension_attribute , user_dataset_condition , reference_dataset_condition )

                # Generate the combined SQL query for the current attribute using the attributes CTE
                combined_query = CTE_fstrings.combined_query_cte_fstring.format(
                    attributes_cte = attributes_cte ,
                    all_selections = all_selections ,
                    table = self.table ,
                    attribute = dimension_attribute ,
                    start = start ,
                    end = end
                )

                print( "Combined Query:" , combined_query )
                combine_df = self.db_handler.ffetch_data( self.database_connection , combined_query , return_df = True )
                [ r , q ] = self.split_combined_df( combine_df )
                ## for pruning
                itr_col = -1
                for measure_attribute in potential_views[ dimension_attribute ]:
                    for aggregate_fn in potential_views[ dimension_attribute ][ measure_attribute ]:
                        itr_view += 1
                        itr_col += 1
                        d = dist_fn_pruning( q[ : , itr_col ] , r[ : , itr_col ] )
                        view_distances.append( d )
                        distance_index_mappings_view[ itr_view ] = ( dimension_attribute , measure_attribute , aggregate_fn )

            ## prune
            pruned_view_indexes = self.prune( view_distances , itr_phase )

            potential_views = self.delete_pruned_views( potential_views , distance_index_mappings_view , pruned_view_indexes ,
            view_distances  )

        suggested_views = self.get_suggested_views( potential_views )
        return suggested_views
    
    def construct_combined_view_query( self , selections , table , dimension_attribute , start , end ):
        return CTE_fstrings.construct_combined_view_query_fstring.format(
            attribute = dimension_attribute ,
            selections = selections ,
            table = table ,
            start = start ,
            end = end
        )

    def filter_df_keyword( self , df , keyword ):
        # Filter columns that contain the keyword and always include the first column (assuming it's the key attribute here)
        filtered_columns =  [ col for col in df.columns if keyword in col ] 
        # Drop duplicates to ensure the first column is not repeated if it contains the keyword
        filtered_columns = list( dict.fromkeys( filtered_columns ) )
        return df[ filtered_columns ]

    def construct_view_query( self , selections , table , condition , dimension_attribute , start , end ):
        view_query = CTE_fstrings.construct_view_query_fstring.format(dimension_attribute, table, selections, condition, start, end)
        return view_query.strip()

    def generate_plot(self, table_query, val_query, val_reference, labels,
    dimension_attribute, aggregate_fn, measure_attribute, label_query, folder_path):
        plt.figure()
        n_groups = table_query.shape[0]
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        # Create bars for the main and reference values
        plt.bar(index, val_query, bar_width,
                alpha=opacity, color='b', label=labels[0])
        plt.bar(index + bar_width, val_reference, bar_width,
                alpha=opacity, color='g', label=labels[1])

        # Set labels, legend, and save the plot
        plt.xlabel(dimension_attribute)
        plt.ylabel('{0}({1})'.format(aggregate_fn, measure_attribute))
        plt.xticks(index + bar_width / 2, label_query, rotation=90)
        plt.legend()
        plt.tight_layout()

        plot_filename = '{0}_{1}_{2}.png'.format(dimension_attribute, measure_attribute, aggregate_fn)
        plt.savefig(os.path.join(folder_path, plot_filename), dpi=300)
        plt.close()


    def visualize( self , views , user_dataset_condition , reference_dataset_condition ,
            labels = None , folder_path = 'visualizations/' ):
        '''

        '''
        if labels == None:
            labels = ['user','Reference']
        for view in views:
            print(view)
            dimension_attribute, measure_attribute, aggregate_fn = view
            # get the query table result
            selection = '__atr__'+' , coalesce('+aggregate_fn + '(' + measure_attribute + '), 0) '
            user_dataset_query = self.construct_view_query(selection,
                    self.table, user_dataset_condition, dimension_attribute, 0, self.num_rows)
            reference_dataset_query = self.construct_view_query(selection,
                    self.table, reference_dataset_condition, dimension_attribute, 0, self.num_rows)
            table_query = np.array( self.db_handler.ffetch_data(self.database_connection, user_dataset_query))
            table_reference = np.array( self.db_handler.ffetch_data(self.database_connection, reference_dataset_query))
            val_query = table_query[ : , 1 ].astype( float )
            label_query = table_query[ : , 0 ]
            val_reference = table_reference[:,1].astype(float)
            a = val_query
            b = val_reference
            self.generate_plot( table_query , val_query , val_reference ,  labels , dimension_attribute , aggregate_fn , measure_attribute ,
            label_query , folder_path )

    def generate_indices_phase( self , num_phases = 10 ):
        '''
        Divides the data into mini-batches (indexes ??)
        and returns the mini-batches (indexes ??) for query.
        With feauture-first grouping ???
        '''
        batch_size = self.num_rows // num_phases  # Using integer division directly
        return [ ( batch_size * i, batch_size * ( i + 1 ) ) for i in range( num_phases ) ]

    def calc_conf_error( self , iter_phase, N_phase = 10 ):
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

    def prune(self, kl_divergences, current_phase, total_phases=10, delta=0.05, top_k=5):
        if current_phase == 1:
            return []

        kl_divergences = np.array(kl_divergences)
        if len(kl_divergences) <= top_k:
            return []  # Return empty or handle differently if there are not enough divergences

        sorted_indices = np.argsort(kl_divergences)[::-1]
        sorted_kl_divergences = kl_divergences[sorted_indices]

        if current_phase == total_phases:
            return sorted_indices[top_k:]

        conf_error = self.calc_conf_error(current_phase, total_phases)
        threshold_kl = sorted_kl_divergences[top_k - 1] - conf_error
        if len(sorted_kl_divergences) > top_k:
            prune_start_index = np.argmax(sorted_kl_divergences[top_k:] + conf_error < threshold_kl) + top_k
            return sorted_indices[prune_start_index:] if prune_start_index < len(sorted_kl_divergences) else []

        return []

if __name__ == '__main__':
    seedb = SeeDB( 'mini_project' , 'postgres' , [ 'fnlwgt' , 'age' , 'capital_gain' , 'capital_loss' , 'hours_per_week' ] ,
    [ 'workclass' , 'education' , 'occupation' , 'relationship' , 'race' , 'native_country' , 'salary' ] , 'census' )
    # recommend_views test
    reference_dataset = "marital_status in (' Divorced', ' Never-married', ' Separated', ' Widowed')"
    user_dataset = "marital_status in (' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse')"
    views = seedb.recommend_views( user_dataset , reference_dataset , 5 )
    labels = [ 'Married' , 'Unmarried' ]
    seedb.visualize( views , user_dataset , reference_dataset ,  [ 'Married' , 'Unmarried' ] ,
    folder_path = 'visualizations/sex/' )

