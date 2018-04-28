import tensorflow as tf
import pandas as pd
import numpy as np
import pdb
# pdb.set_trace()
import math

STATES = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut",
          "District of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois",
          "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota",
          "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire",
          "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico",
          "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands",
          "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
HIDDEN_UNITS = [10, 10, 10]
BATCH_SIZE = 100
STEPS = 1000

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
  pdb.set_trace()
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_cols, hidden_units=HIDDEN_UNITS, n_classes=4)

  classifier.train(
    input_fn=lambda:train_input(train_data, train_labels),
    steps=STEPS
  )

  eval_result = classifier.evaluate(input_fn=lambda:eval_input(validation_data, validation_labels))

  print('\nValidation set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

  predictions = classifier.predict(input_fn=lambda:eval_input(test_data, labels=None))

  template = ('\nPredicted "{}" with probability of ({:.1f}%), correct value is "{}"')
  for guess, correct in zip(predictions, train_labels):
    class_id = guess['class_ids'][0]
    prob = guess['probabilities'][class_id]

    print(template.format(class_id, prob, correct))

  print "Success...ish"



def load_data(path):
  working_frame = pd.read_csv(path)
  working_frame.drop(['Department', 'Year', 'Population', 'Randomizer'], axis = 1, inplace = True)
  fifteen_percent = int(math.floor(working_frame.shape[0] * .15))

  test_data = working_frame.head(fifteen_percent)
  working_frame.drop(working_frame.index[:fifteen_percent], inplace=True)

  validation_data = working_frame.head(fifteen_percent)
  working_frame.drop(working_frame.index[:fifteen_percent], inplace=True)

  test_label = test_data.pop('Pop_Quartile')
  validation_label = validation_data.pop('Pop_Quartile')
  training_label = working_frame.pop('Pop_Quartile')

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
  main()

