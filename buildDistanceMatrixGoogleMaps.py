# -*- coding: utf-8 -*-
"""BuildDistanceMatrix.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TSH75tLs6kOelu3mILXp-c-2PVvhPFIJ
"""

import requests as rq
import json as js
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="POapp")

## API da google para obter as distâncias entre duas cidades
## Documentação: https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes
##               https://developers.google.com/maps/documentation/routes/compute_route_directions
def getDistance(cidade1,cidade2):
  headers = {
      'Content-Type': 'application/json',
      'X-Goog-Api-Key': '',#Entrar com a chave de acesso da API
      'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters',
  }

  json_data = {
      'origin': {
          'address': cidade1
      },
      'destination': {
          'address': cidade2
      },
      'travelMode': 'DRIVE',
      'routingPreference': 'TRAFFIC_UNAWARE',
      'computeAlternativeRoutes': False,
      'routeModifiers': {
          'avoidTolls': False,
          'avoidHighways': False,
          'avoidFerries': False,
      },
      'languageCode': 'pt-BR',
      'units': 'METRIC',
  }

  response = rq.post('https://routes.googleapis.com/directions/v2:computeRoutes', headers=headers, data=js.dumps(json_data))
  try:
    content=response.json()
    distance=content['routes'][0]['distanceMeters']
  except Exception:
    print(cidade1+" -> "+cidade2)
    print("Status code: "+str(response.status_code))
    print(response.content)
    raise Exception
  else:
    return distance


dados=pd.read_csv('/content/CidadesMG.csv', index_col='ID')
cidades=dados['Cidade']

for i in range(1,854):#cidade inicial, cidade final+1
  currentColunm=[9999999] * (len(dados.index))
  try:
    for j in range(0,(i-1)):
      currentColunm[j]=getDistance(cidades[i],cidades[j+1])
  except Exception:
    print("i= "+str(i))
    print("j= "+str(j))
    dados.to_csv('/content/dados.csv',',')
    raise Exception
  else:
    dados[str(i)]=currentColunm

dados.to_csv('/content/dados.csv',',')