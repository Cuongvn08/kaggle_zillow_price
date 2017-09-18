# -*- coding: utf-8 -*-

from enum import Enum

## do setting
class Settings(Enum):
    train_path = 'C:/data/kaggle/zillow_price/train_2016_v2.csv'
    properties_path = 'C:/data/kaggle/zillow_price/properties_2016.csv'
    zillow_data_dictionary_path = 'C:/data/kaggle/zillow_price/zillow_data_dictionary.csv'
    
    def __str__(self):
        return self.value
        
## process data
def process_data():
    pass

## build model
def build_model():
    pass

## generate submission
def generate_submission():
    pass

## main
def main():
    pass

if __name__ == "__main__":
    main()
    print('The end.')
