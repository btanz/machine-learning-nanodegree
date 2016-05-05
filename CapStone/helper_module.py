""" This module contains helper functions used in the
machine learning engineer nanodegree capstone project.
"""
import numpy as np
import pandas as pd
import urllib.request, urllib.parse, json
from collections import OrderedDict
import math

#GOOGLE_GEOCODE_API_KEY = 'AIzaSyCEH-nDNaGfDswyrB10-J4K_IQEITlYGsc'   # simplyfi
GOOGLE_GEOCODE_API_KEY = 'AIzaSyAp9A4SOuRY3bLGPy45T1eSYrgQ-Te0z_k'  #benjamintanz@gmx.de

GOOGLE_GEOCODE_URI = 'https://maps.googleapis.com/maps/api/geocode/json'
OPENSTREETMAP_GEOCODE_URI = 'http://nominatim.openstreetmap.org/search/'

def load_transaction_data():
    """Loads all transactions data and returns a dataframe containing
    the data."""
    data2010 = pd.read_csv('./data/PPR-2010.csv', encoding='latin_1')
    data2011 = pd.read_csv('./data/PPR-2011.csv', encoding='latin_1')
    data2012 = pd.read_csv('./data/PPR-2012.csv', encoding='latin_1')
    data2013 = pd.read_csv('./data/PPR-2013.csv', encoding='latin_1')
    data2014 = pd.read_csv('./data/PPR-2014.csv', encoding='latin_1')
    data2015 = pd.read_csv('./data/PPR-2015.csv', encoding='latin_1')
    return pd.concat([data2010, data2011, data2012, data2013, data2014, data2015])


def load_income_data():
    """Loads the county-level income data and extracts disposable income per person
        and income growth per county."""
    income_data = pd.read_csv('./data/income_data.csv', encoding='latin_1', names=['region', 'feature', '2010', '2011', '2012', '2013', '2014'])
    income_data = income_data[income_data.feature=='Disposable Income per Person (Euro)']
    income_data['income_growth_5y'] = (income_data['2014'] / income_data['2010'] - 1) * 100
    income_data = income_data[['region', '2014', 'income_growth_5y']]
    income_data.columns = [['region', 'income_2014', 'income_growth_5y']]
    # standardize some fields
    income_data.loc[income_data['region']=='Cork City and County', 'region'] = 'Cork'
    income_data.loc[income_data['region']=='Galway City and County', 'region'] = 'Galway'
    income_data.loc[income_data['region']=='Limerick City and County', 'region'] = 'Limerick'
    income_data.loc[income_data['region']=='Waterford City and County', 'region'] = 'Waterford'
    income_data.loc[income_data['region']=='South Tipperary', 'region'] = 'Tipperary'
    return income_data


def geocode_data(df, col, limit=61500):
    """Geocode the address data in column 'col' if dataframe 'df'. Skip if values are
    already geocoded."""
    i = 1
    for (index, row) in df.iterrows():
        i += 1
        # temp
        if i < 58900:
            continue
        if i > limit:
            break
        if not np.isnan(row['lat']):
            continue
        print('working...')
        values = {'address' : row[col], 'key' : GOOGLE_GEOCODE_API_KEY}
        data = urllib.parse.urlencode(values)
        full_url = GOOGLE_GEOCODE_URI + '?' + data
        response = urllib.request.urlopen(full_url)
        jsonResponse = json.loads(response.readall().decode('utf-8'))
        try:
            df.at[index, 'lat'] = jsonResponse['results'][0]['geometry']['location']['lat']
            df.at[index, 'lon'] = jsonResponse['results'][0]['geometry']['location']['lng']
        except IndexError:
            print('An index error occured')

    return df


def distance(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between any two GPS coordinates (lat1, lon1) and
    (lat2, lon2)."""

    # approximate radius of earth in km
    R = 6373.0

    # convert to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance



def map_closest_police(df_data, df_crime_data):
    """Return df_data with two additional columns containing id
        and distance of the closest police station. Note that this
        may take a while, as the algorithm is len(df_data)*len(df_crime_data). """
    df_data['ClosestPoliceId'] = None
    df_data['ClosestPoliceDist'] = None
    for (index_property, series_property) in df_data.iterrows():
        closest_id = None
        closest_dist = 99999

        if np.isnan(series_property.lat) or np.isnan(series_property.lon):
            continue
        for (index_crime, series_crime) in df_crime_data.iterrows():
            dist = distance(series_property.lat, series_property.lon, series_crime.lat, series_crime.lon)
            if dist < closest_dist:
                closest_dist = dist
                closest_id = index_crime

        df_data.at[index_property, 'ClosestPoliceId'] = closest_id
        df_data.at[index_property, 'ClosestPoliceDist'] = closest_dist

    return df_data



def map_population_centers(df_data):
    """Return df_data with two additional columns containing id
        and distance of the closest population center.  """

    pop_centers = pd.read_csv('./data/pop_centers.csv', encoding='utf8')

    df_data['ClosestCenterId'] = None
    df_data['ClosestCenterDist'] = None
    for (index_property, series_property) in df_data.iterrows():
        closest_id = None
        closest_dist = 99999

        if np.isnan(series_property.lat) or np.isnan(series_property.lon):
            continue
        for (index_center, series_center) in pop_centers.iterrows():
            dist = distance(series_property.lat, series_property.lon, series_center.Lat, series_center.Lon)
            if dist < closest_dist:
                closest_dist = dist
                closest_id = index_center

        df_data.at[index_property, 'ClosestCenterId'] = closest_id
        df_data.at[index_property, 'ClosestCenterDist'] = closest_dist

    return df_data