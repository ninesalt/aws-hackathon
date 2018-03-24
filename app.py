from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from flask import Flask
from flask_cors import CORS
from flask import request, jsonify

directory = './rdbms-comp/rdbms-tab-all/'
locationsFile = directory + 'Coords.txt'
mineralsFile = directory + 'Materials.txt'
commoditiesFile = directory + 'Production_detail.txt'
conditionsFile = directory + 'Alteration.txt'

# parse files
locations = pd.read_csv(locationsFile, sep="\t", low_memory=False)[
    ['dep_id', 'lat_dec', 'lon_dec']]
commodities = pd.read_csv(commoditiesFile, sep="\t", low_memory=False)[
    ['dep_id', 'commod']]
minerals = pd.read_csv(mineralsFile, sep="\t", low_memory=False)[
    ['dep_id', 'material']]
conditions = pd.read_csv(conditionsFile, sep="\t", low_memory=False)[
    ['dep_id', 'alterat_text']]

# join tables by id
joined = pd.merge(locations, minerals, left_on="dep_id",
                  right_on="dep_id", how="left")
# joined = pd.merge(joined, conditions, left_on="dep_id", right_on="dep_id", how="left")
joined = pd.merge(joined, commodities, left_on="dep_id",
                  right_on="dep_id", how="left")
joined = joined.dropna(axis=0, how='any')  # drop NaNs

# remove uncommon commodities
value_counts = joined.commod.value_counts()
avg = np.average(value_counts)
# most common commodities as strings
filtered_coms = value_counts[value_counts > avg].index.tolist()

# replace unwanted commodities in table
joined.commod.loc[~joined.commod.isin(filtered_coms)] = 'Other'
filtered_coms.append('Other')

# label commodity classes
lb = preprocessing.LabelEncoder()
commodity_classes = lb.fit_transform(
    joined.commod)  # commodity classes as numbers

# remove uncommon minerals
value_counts = joined.material.value_counts()
# avg = np.mean(value_counts)
# filtered_minerals = value_counts[value_counts > avg*15].index.tolist() #most common commodities as strings
filtered_minerals = value_counts[:5].index.tolist()

# replace unwanted minerals in table
joined.material.loc[~joined.material.isin(filtered_minerals)] = 'Other'
filtered_minerals.append('Other')

# label mineral classes
lb2 = preprocessing.LabelEncoder()
mineral_classes = lb2.fit_transform(
    joined.material)  # mineral classes as numbers

# cv = CountVectorizer(strip_accents='ascii', stop_words='english', max_features=150)
# conditions_vec = cv.fit_transform(joined.alterat_text)    # main conditions transformed vector

processed_table = joined.copy()
processed_table.commod = commodity_classes
processed_table.material = mineral_classes


def create_model(features, output, destination, target,
                 override_feature=None, processed_table=processed_table):

    model_input = processed_table[features] if override_feature is None else override_feature
    model_output = processed_table[output]
    x_train, x_test, y_train, y_test = train_test_split(model_input, model_output,
                                                        test_size=.3, random_state=17)
    model = RandomForestClassifier(n_jobs=-1)
    print('Training...')
    model.fit(x_train, y_train)
    print('Training done. \nRunning predictions...\n')
    predictions = model.predict(x_test)

    print(destination, '\n')
    print(classification_report(y_test, predictions, target_names=target))
    print('Average accuracy: {}%'.format(
        np.around(accuracy_score(y_test, predictions) * 100, 2)))
    print('\n\n##################################################################################\n')
    return model


# create models given features and output
m1 = create_model(['lat_dec', 'lon_dec'], 'material',
                  'Location -> Minerals', filtered_minerals)

m2 = create_model(['lat_dec', 'lon_dec'], 'commod',
                  'Location -> Commodity', filtered_coms)


def predict(model_input, model_type, m1=m1, m2=m2, filtered_minerals=filtered_minerals,
            commodity_classes=commodity_classes):

    model_input = np.array(model_input)
    model_input = np.reshape(model_input, (1, -1))

    if model_type == 'mineral':
        prediction = m1.predict_proba(model_input)[0]
        return pd.DataFrame({'{}'.format(model_type): filtered_minerals, 'Probability (%)': prediction * 100})

    if model_type == 'commodity':
        prediction = m2.predict_proba(model_input)[0]
        return pd.DataFrame({'{}'.format(model_type): filtered_coms, 'Probability (%)': prediction * 100})

# other models

# m3 = create_model(['lat_dec', 'lon_dec', 'material'], 'commod', 'Location + mineral -> Commodity', filtered_coms)
# m4 = create_model(['material'], 'commod', 'Mineral -> Commodity', filtered_coms)
# m1 = create_model(['commod'], 'material', 'Commodity -> Minerals', filtered_minerals)
# m5 = create_model(None, 'commod', 'Conditions -> commodity', filtered_coms, conditions_vec)


app = Flask(__name__)
CORS(app)


@app.route('/isAlive')
def index():
    return "true"


@app.route('/predict/', methods=['GET'])
def get_prediction():

    coord_lat = float(request.args.get('lat'))
    coord_long = float(request.args.get('long'))
    features = [coord_lat, coord_long]

    prediction_m1 = predict(features, 'mineral').to_json()
    prediction_m2 = predict(features, 'commodity').to_json()

    preds = {'minerals': prediction_m1, 'coms': prediction_m2}
    return str(preds)


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
