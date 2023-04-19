# Ambiguous Entity Oriented Targeted Document Detection
## Dependencies

* Compatible with Python 3.7
* Dependencies can be installed using requirements.txt


### Datasets
We construct four labeled datasets for the targeted document detection task, i.e., `Wiki-100`, `Wiki-200`, `Wiki-300`, and `Web-Test`. The former three datasets are
constructed from Wikipedia and the last dataset is constructed from Web documents.

* The four labeled datasets `Wiki-100`, `Wiki-200`, `Wiki-300`, and `Web-Test` are placed in the `datasets` folder. Please unzip `Wiki100.zip`, `Wiki200.zip`, `Wiki300.zip`, and `Web_Test.zip` under `datasets/`.

### Usage

##### Run the main code (**GADE**):

* python GADE_main_Wiki.py --model_name GADE_100 --data_type Wiki100

* python GADE_main_Wiki.py --model_name GADE_200 --data_type Wiki200

* python GADE_main_Wiki.py --model_name GADE_300 --data_type Wiki300

##### Run the main code for the base architecture **GADE-local**:

* python GADE_local_main_Wiki.py --model_name GADE_local_100 --data_type Wiki100

* python GADE_local_main_Wiki.py --model_name GADE_local_200 --data_type Wiki200

* python GADE_local_main_Wiki.py --model_name GADE_local_300 --data_type Wiki300

##### Test the model's performance on Web-Test dataset:

-- For **GADE**:

* python GADE_main_Web_Test.py

-- For **GADE-local**:

* python GADE_local_main_Web_Test.py

### Contact
