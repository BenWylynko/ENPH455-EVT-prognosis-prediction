import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_ids_labels():
    a_file = open(Path(Path(__file__).parents[0], "patient_id_label_dict.pkl"), "rb")
    ids_labels = pickle.load(a_file)
    return ids_labels

def gen_split_3(sdir, X, ids, labels, ids_labels, val_test_size):
    #train, val + test splits
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        X, ids, stratify=labels, test_size=val_test_size, random_state=42) 
    #get labels for next stratification
    y_val_test_labels = [ids_labels[id_] for id_ in y_val_test]
    #val, test splits
    x_test, x_val, y_test, y_val = train_test_split(
        x_val_test, y_val_test, stratify=y_val_test_labels, test_size=0.5, random_state=80) 
    #save splitting
    print(f"Splitting {len(labels)} images into train: {len(y_train)} samples, val: {len(y_val)} samples, test: {len(y_test)} samples")
    split_ids = {'train': y_train, 'val': y_val, 'test': y_test}
    #always save to utils dir
    a_file = open(Path(sdir, f"dataset_split_3_{val_test_size}.pkl"), "wb")
    pickle.dump(split_ids, a_file)
    a_file.close()

    return x_train, y_train, x_val, y_val, x_test, y_test

def gen_split_2(sdir, X, ids, labels, ids_labels, test_size):
    #train, val + test splits
    x_train, x_test, y_train, y_test = train_test_split(
        X, ids, stratify=labels, test_size=test_size, random_state=42) 
    #save splitting
    print(f"Splitting {len(labels)} images into train: {len(y_train)} samples, test: {len(y_test)} samples")
    split_ids = {'train': y_train, 'test': y_test}
    #always save to utils dir
    sfile = Path(sdir, f"dataset_split_2_{test_size}.pkl")
    a_file = open(sfile, "wb")
    pickle.dump(split_ids, a_file)
    a_file.close()
    return x_train, y_train, x_test, y_test

def gen_split_2_stratified(sdir, LOV, ids, labels, ids_labels, test_size):
    """Stratify based on both label and LOV"""
    LOV_labels = [str(lov) + '_' + str(label) for lov, label in zip(LOV, labels)]
    #train, val + test splits
    _, _, y_train, y_test = train_test_split(
        LOV, ids, stratify=LOV_labels, test_size=test_size, random_state=42) 
    #save splitting
    print(f"Splitting {len(labels)} images into train: {len(y_train)} samples, test: {len(y_test)} samples")
    split_ids = {'train': y_train, 'test': y_test}
    #always save to utils dir
    a_file = open(Path(sdir, f"dataset_split_2_{test_size}.pkl"), "wb")
    pickle.dump(split_ids, a_file)
    a_file.close()
    return y_train, y_test

def load_split_3(val_test_size):
    #load labels from pickle
    a_file = open(Path(Path(__file__).parents[0], f"dataset_split_3_{val_test_size}.pkl"), "rb")
    split_ids = pickle.load(a_file)
    y_train = split_ids['train']
    y_val = split_ids['val']
    y_test = split_ids['test']
    return y_train, y_val, y_test

def load_split_2(test_size):
    #load labels from pickle
    file_ = Path(Path(__file__).parents[0], f"dataset_split_2_{test_size}.pkl")
    a_file = open(file_, "rb")
    split_ids = pickle.load(a_file)
    y_train = split_ids['train']
    y_test = split_ids['test']
    return y_train, y_test

