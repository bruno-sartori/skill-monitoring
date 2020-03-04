## BUILD
conda env create -f environment.yml
conda activate skill-monitoring

## RUN
PYTHON_ENV=development python run.py

## USAGE
press enter to save screenshot!
press esc to quit


## KERAS RECOGNIZER
cd monitoring/modules/keras_recognizer
python setupData.py
python trainClassifier.py
python predict.py
