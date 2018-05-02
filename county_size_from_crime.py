import tensorflow as tf
import pandas as pd
import numpy as np
import pdb
import math

STATES = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut",
          "District of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois",
          "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota",
          "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire",
          "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico",
          "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands",
          "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

# Options to passed to classifier.  Set globally for ease of adjustment.
HIDDEN_UNITS = [10]
BATCH_SIZE = 10
STEPS = 10000

def main():

  (train_data, train_labels), (validation_data, validation_labels), (test_data, test_labels) = load_data('county_crime.csv')

  # Create feature columns
  feature_cols = []
  for key in train_data.keys():
    if (key == 'State'):
      temp = tf.feature_column.categorical_column_with_vocabulary_list('State', STATES)
      feature_cols.append(tf.feature_column.embedding_column(temp, 51))
    else:
      feature_cols.append(tf.feature_column.numeric_column(key=key))

  # Instantiate model
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_cols, hidden_units=HIDDEN_UNITS, n_classes=4)

  # train model
  classifier.train(
    input_fn=lambda:train_input(train_data, train_labels),
    steps=STEPS
  )

  # evaluate trained model
  eval_result = classifier.evaluate(input_fn=lambda:eval_input(test_data, test_labels))
  print('\nTest set accuracy: {accuracy:0.2f}%\n'.format(**eval_result))

  # make some predictions
  predictions = classifier.predict(input_fn=lambda:eval_input(validation_data, labels=None))

  # tally percentage of correct predictions
  swings = 0
  hits = 0
  for guess, correct in zip(predictions, train_labels):
    swings += 1
    class_id = guess['class_ids'][0]
    prob = guess['probabilities'][class_id]
    if(class_id == correct and prob > .7):
      hits += 1


  print '\nValidation set accuracy: ' + "{0:.2f}".format(float(hits)/swings * 100) + '%\n'
# end main


def load_data(path):
  # load data and drop undesired columns
  working_frame = pd.read_csv(path)
  working_frame.drop(['Department', 'Year', 'Population', 'Randomizer'], axis = 1, inplace = True)

  # randomize the rows
  working_frame = working_frame.reindex(np.random.permutation(working_frame.index))

  # calulate how records is 15% of total records
  fifteen_percent = int(math.floor(working_frame.shape[0] * .15))

  # pare off 15% of records for testing
  test_data = working_frame.head(fifteen_percent)
  working_frame.drop(working_frame.index[:fifteen_percent], inplace=True)

  # pare off 15% of records for validation
  validation_data = working_frame.head(fifteen_percent)
  working_frame.drop(working_frame.index[:fifteen_percent], inplace=True)

  # pop off the Pop_Quartile column from each dataset to be its label
  test_label = test_data.pop('Pop_Quartile')
  validation_label = validation_data.pop('Pop_Quartile')
  training_label = working_frame.pop('Pop_Quartile')

  # return 3 pairs formatted as (features, labels)
  return (working_frame, training_label), (validation_data, validation_label), (test_data, test_label)

def train_input(features, labels, batch_size=BATCH_SIZE):
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
  dataset = dataset.shuffle(100000).repeat().batch(batch_size)

  return dataset

def eval_input(features, labels, batch_size=BATCH_SIZE):
  features=dict(features)
  if labels is None:
    inputs = features
  else:
    inputs = (features, labels)

  dataset = tf.data.Dataset.from_tensor_slices(inputs)
  dataset = dataset.batch(batch_size)

  return dataset

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()

