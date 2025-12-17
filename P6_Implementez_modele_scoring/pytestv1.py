import pandas as pd
import pathlib

def nbcolonne():
    data = pd.read_csv(r'C:\Users\herbett\OneDrive - LEMKEN GmbH & Co.KG\Documents\Data_Science_OC\Implementez_un_modele_de_scoring\X_train.csv')
    nb_colonne = data.shape[1]
    return nb_colonne

def test_nb_colonne():
    assert nbcolonne() == 238

def are_all_columns_numeric():
    data = pd.read_csv(r'C:\Users\herbett\OneDrive - LEMKEN GmbH & Co.KG\Documents\Data_Science_OC\Implementez_un_modele_de_scoring\X_train.csv')
    numeric_columns = data.slect_dtypes(include=['number']).columns
    return len(numeric_columns) == len(data.columns)

def test_are_all_columns_numeric():
    assert are_all_columns_numeric()

def is_target_exist():
    data = pd.read_csv(r'C:\Users\herbett\OneDrive - LEMKEN GmbH & Co.KG\Documents\Data_Science_OC\Implementez_un_modele_de_scoring\X_train.csv')

    assert 'TARGET' in data.columns, "La variabl 'TARGET' n'existe pas dans le DataFrame."

def test_model():
    dir_path = pathlib.Path(r'C:\Users\herbett\OneDrive - LEMKEN GmbH & Co.KG\Documents\Data_Science_OC\Implementez_un_modele_de_scoring')

    file_name = 'best_lgbm_model.pkl'

    file_path = dir_path / file_name

    assert file_path.exists(), f"Le fichier {file_name} n'existe pas dans le répertoire {dir_path}."
