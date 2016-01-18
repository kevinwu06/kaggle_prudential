import pandas as pd 
import os 

class Munger():
    def __init__(self):
        
        # global variables
        self.columns_to_drop = ['Id', 'Response']
        basedir = '~/Documents/Github/kaggle_prudential'
        basedir = os.path.expanduser(basedir)
        
        train = pd.read_csv(os.path.join(basedir,"input/train.csv"))
        test = pd.read_csv(os.path.join(basedir,"input/test.csv"))
        
        # combine train and test
        self.all_data = train.append(test)
               
    def engineer_features(self):
        self.all_data['Product_Info_2_char'] = self.all_data.Product_Info_2.str[1]
        self.all_data['Product_Info_2_num'] = self.all_data.Product_Info_2.str[2]
    '''    
        discrete = ['Medical_History_1','Medical_History_10', 'Medical_History_15', 
                    'Medical_History_24', 'Medical_History_32']
        
        continuous = ['Product_Info_4','Ins_Age','Ht','Wt','BMI','Employment_Info_1',
                      'Employment_Info_4','Employment_Info_6','Insurance_History_5',
                      'Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5']
        
        
        # variable transformations: squaring and cubing all continuous variables
        for i, f in enumerate(self.all_data[continuous]):
            self.all_data[(continuous[i]+'_sq')] = self.all_data[f]**2
        
        for i, f in enumerate(self.all_data[continuous]):
            self.all_data[(continuous[i]+'_cub')] = self.all_data[f]**3
            
        # squaring Ht & Wt
        self.all_data['Ht_sq'] = self.all_data['Ht']**2
        self.all_data['Wt_sq'] = self.all_data['Wt']**2
        self.all_data = self.all_data.drop(['Ht','Wt'], axis=1)
    '''  
    def one_hot_encode(self):        
        categorical = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 
                       'Product_Info_5', 'Product_Info_6', 'Product_Info_7',
                       'Product_Info_2_char', 'Product_Info_2_num', 
                       'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 
                       'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 
                       'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 
                       'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 
                       'Insurance_History_3', 'Insurance_History_4', 
                       'Insurance_History_7', 'Insurance_History_8', 
                       'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 
                       'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 
                       'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 
                       'Medical_History_9', 'Medical_History_11',
                       'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 
                       'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 
                       'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 
                       'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 
                       'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 
                       'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 
                       'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 
                       'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 
                       'Medical_History_39', 'Medical_History_40', 'Medical_History_41',]

        for f in categorical:
            all_data_dummy = pd.get_dummies(self.all_data[f], prefix=f)
            self.all_data = self.all_data.drop([f], axis=1)
            self.all_data = pd.concat((self.all_data, all_data_dummy), axis=1)
 
    def missing_val_handling(self):         
        print('Eliminate missing values')    
        # Use -1 for any others
        self.all_data.fillna(-1, inplace=True)
        # fix the dtype on the label column
        self.all_data['Response'] = self.all_data['Response'].astype(int)
    
    def data_split(self):
        # split train and test
        train = self.all_data[self.all_data['Response']>0].copy()
        test = self.all_data[self.all_data['Response']<1].copy()
        return train, test
        
def clean_data():
    M = Munger()
    M.engineer_features()
    M.one_hot_encode()
    M.missing_val_handling()
    return M