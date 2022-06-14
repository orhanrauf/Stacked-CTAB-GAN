import pandas as pd
import time
# Used for pre/post-processing of the input/generated data
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_first_layer import CTABGANFirstLayer

# Model class for the CTABGANSynthesizer

import warnings

from model.synthesizer.ctabgan_second_layer import CTABGANSecondLayer
from model.synthesizer.stacked_condvec_factory import StackedCondvecFactory
from model.synthesizer.transformer import DataTransformer

warnings.filterwarnings("ignore")

class StackedCTABGAN():

    """
    Generative model training class based on the CTABGANSynthesizer model

    Variables:
    1) raw_csv_path -> path to real dataset used for generation
    2) test_ratio -> parameter to choose ratio of size of test to train data
    3) categorical_columns -> list of column names with a categorical distribution
    4) log_columns -> list of column names with a skewed exponential distribution
    5) mixed_columns -> dictionary of column name and categorical modes used for "mix" of numeric and categorical distribution 
    6) integer_columns -> list of numeric column names without floating numbers  
    7) problem_type -> dictionary of type of ML problem (classification/regression) and target column name
    8) epochs -> number of training epochs

    Methods:
    1) __init__() -> handles instantiating of the object with specified input parameters
    2) fit() -> takes care of pre-processing and fits the CTABGANSynthesizer model to the input data 
    3) generate_samples() -> returns a generated and post-processed sythetic dataframe with the same size and format as per the input data 

    """

    def __init__(self,
                 raw_csv_path = "Real_Datasets/Adult.csv",
                 intermediate_data_path = "Intermediate_Datasets/Adult_first_layer.csv",
                 intermediate_cond_path = "Intermediate_Datasets/Adult_condvecs.csv",
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'},
                 epochs = 1,
                 batch_size=500):

        self.__name__ = 'StackedCTABGAN'
              
        self.first_synthesizer = CTABGANFirstLayer(epochs = epochs)
        self.second_synthesizer = CTABGANSecondLayer(epochs = epochs)
        self.raw_df = pd.read_csv(raw_csv_path)
        self.intermediate_data_path = intermediate_data_path
        self.intermediate_cond_path = intermediate_cond_path
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.batch_size = batch_size
        
    def fit(self):
        
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,
                                  self.integer_columns,self.problem_type,self.test_ratio)
        
        self.first_synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], 
                             mixed = self.data_prep.column_types["mixed"],type=self.problem_type)
        end_time = time.time()
        
        print('Finished training in',end_time-start_time," seconds.")
        
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,
                                  self.integer_columns,self.problem_type,self.test_ratio)
        
        self.data_transformer = DataTransformer(train_data=self.data_prep.df, 
                                           categorical_list = self.data_prep.column_types["categorical"], 
                                           mixed_dict = self.data_prep.column_types["mixed"])
        self.data_transformer.fit()
        self.train_data = self.data_transformer.transform(self.data_prep.df.values)
        self.stacked_condvec_factory = StackedCondvecFactory(self.train_data, self.data_transformer.output_info, self.batch_size)
        intermediate_data = self.first_synthesizer.sample_train(len(self.raw_df), self.stacked_condvec_factory)
        
        pd.DataFrame(intermediate_data).to_csv("Adult_intermediate_data.csv")
        
        self.second_synthesizer.fit(train_data=self.data_prep.df, stacked_condvec_factory=self.stacked_condvec_factory,
                                intermediate_data=intermediate_data, categorical = self.data_prep.column_types["categorical"], 
                                mixed = self.data_prep.column_types["mixed"],type=self.problem_type)
        
        
        print(intermediate_data)
        
    


    def generate_samples(self):
        
        stacked_condvec_factory = StackedCondvecFactory(self.train_data, self.data_transformer.output_info, self.batch_size)
        intermediate_data = self.first_synthesizer.sample(len(self.raw_df), stacked_condvec_factory)
        sample = self.second_synthesizer.sample(intermediate_data, len(self.raw_df), stacked_condvec_factory)
        sample_df = self.data_prep.inverse_prep(sample)
        return sample_df
