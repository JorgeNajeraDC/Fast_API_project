import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import jwt
from dotenv import load_dotenv, find_dotenv
from typing import Optional
import json
import jwt
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
import boto3
import json
import jwt
import numpy as np
import os
import pandas as pd
import pyodbc
import pytz
import requests
import smtplib
import urllib.parse
import urllib.request
import uuid
from ast import literal_eval
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
from email.message import EmailMessage
from math import sin, cos, sqrt, atan2, radians
import uvicorn
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

def dist(i, j):
    lat1 = radians(i[0])
    lon1 = radians(i[1])
    lat2 = radians(j[0])
    lon2 = radians(j[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return 6373 * c * 1000

def cal_estaticas(cp, valor_m2, conn):
    cursor = conn.cursor()

    sql = f"""  

                SELECT CodigoPostal,
                       Score_Transacciones_SHF,
                       Score_Plusvalia,
                       Score_Servicios,
                       Plusvalia,
                       TimeOnMarket,
                       Score_TimeOnMarket,
                       m2_venta_Q1_Municipio,
                       m2_venta_Q2_Municipio,
                       m2_venta_Q3_Municipio,
                       m2_venta_Q4_Municipio

                FROM [ai360cloudprod].[dbo].[vwFact_Score_CP_Avaluo]

                WHERE CodigoPostal = '{cp}'

                AND [Plusvalia] IS NOT NULL

           """

    cursor.execute(sql)

    data = cursor.fetchall()
    columns = cursor.description
    data = [{columns[index][0]: column for index, column in enumerate(value)} for value in data]
    data = pd.DataFrame(data)

    if len(data) >= 0:

        c_precio = 5 if valor_m2 <= data['m2_venta_Q1_Municipio'].values[0] else (
            4 if valor_m2 <= data['m2_venta_Q2_Municipio'].values[0] else (
                3 if valor_m2 <= data['m2_venta_Q3_Municipio'].values[0] else (
                    2 if valor_m2 <= data['m2_venta_Q4_Municipio'].values[0] else (
                        1 if valor_m2 > data['m2_venta_Q4_Municipio'].values[0] else (np.nan)))))

        if (type(data['TimeOnMarket'].values[0]) == np.float64 or type(data['TimeOnMarket'].values[0]) == float):

            data['TimeOnMarket'].values[0] = data['TimeOnMarket'].values[0].round(1)

        else:

            data['TimeOnMarket'].values[0] = 'Codigo Postal sin ofertas'

        data = {'CodigoPostal': data['CodigoPostal'].values[0],
                'Plusvalia': data['Plusvalia'].values[0],
                'TimeOnMarket': data['TimeOnMarket'].values[0],
                'Score_TimeOnMarket': int(data['Score_TimeOnMarket'].values[0]),
                'Score_Transacciones_SHF': int(data['Score_Transacciones_SHF'].values[0]),
                'Score_Plusvalia': int(data['Score_Plusvalia'].values[0]),
                'Score_Servicios': int(data['Score_Servicios'].values[0]),
                'Score_PrecioM2': c_precio}

        return data

    else:

        return 'Codigo Postal Invalido'

def bd_cal_nse(cp, digitos, conn):
    cp = cp[:digitos]

    cursor = conn.cursor()

    sql = f"""  

                SELECT Latitud,
                       Longitud,
                       NSE,
                       CalNSE

                FROM [ai360cloudprod].[dbo].[FactAvaluoNSE]

                WHERE NSE >0 AND LEFT(CodigoPostal, {str(digitos)}) = '{str(cp)}'

           """

    cursor.execute(sql)
    data = cursor.fetchall()

    columns = cursor.description
    data = [{columns[index][0]: column for index, column in enumerate(value)} for value in data]
    data = pd.DataFrame(data)

    if len(data) >= 0:

        return data

    else:

        return []

def get_clase_nse_cal(latitud, longitud, cp, acabados, amenidades, conn):
    flotante = bd_cal_nse(cp, 5, conn)

    if len(flotante) == 0:

        flotante = bd_cal_nse(cp, 4, conn)

        if len(flotante) == 0:
            flotante = bd_cal_nse(cp, 3, conn)

    for a, b, c in flotante[['Latitud', 'Longitud']].itertuples():
        flotante.at[a, 'distancia'] = dist([latitud, longitud], [b, c])

    nse = flotante.sort_values('distancia').head(1)['NSE'].values[0]

    cal_nse = int(flotante.sort_values('distancia').head(1)['CalNSE'].values[0])

    clase = 6 if ((acabados == 3) | (amenidades >= 4)) else (
        5 if ((acabados == 2) | (amenidades == 3)) else (
            4 if ((acabados == 1) | (amenidades == 2)) else (
                3 if ((acabados == 0) & (amenidades <= 1)) else (
                    np.nan))))

    clase = 4 if ((nse > 195) & (clase == 3)) else (
        5 if ((nse > 250) & ((clase == 3) | (clase == 4))) else (
            6 if (nse > 280) else (
                clase)))

    dic_clase_nse_cal = {'nse': nse,
                         'cal_nse': cal_nse,
                         'clase': clase}

    del flotante

    return dic_clase_nse_cal

def SP_promedio(cp, tipo, clase, conser, sup, conn):
    cursor = conn.cursor()

    sql = f"""

                EXEC [dbo].[spCombinacionesPromedio_Top_1]
                @cp = '{cp}',
                @tipo = {tipo},
                @clase = {clase},
                @conser = {conser},
                @sup = {sup}

          """

    cursor.execute(sql)

    data = cursor.fetchall()
    columns = cursor.description
    data = [{columns[index][0]: column for index, column in enumerate(value)} for value in data]
    data = pd.DataFrame(data)

    if data.empty:
        validador = 1
        return validador

    else:
        dic_comb = {'prom_m2_venta': int(data['prom_m2_venta'].values[0]),
                    'cuenta': data['cuenta'].values[0],
                    'IdCombinacion': data['IdCombinacion'].values[0],
                    'combinacion': data['combinacion'].values[0],
                    'dif': data['MaxMin'].values[0]}
        del data
        return dic_comb

def coordenadas(direccion):
    serviceurl = 'https://maps.googleapis.com/maps/api/geocode/json?'
    ApiKey = 'key=AIzaSyAMSZbv76mXJxCHkGZVPJV2wMFmLH0W1mE'

    url = serviceurl + urllib.parse.urlencode({'address': direccion}) + '&' + ApiKey
    uh = urllib.request.urlopen(url)
    data = uh.read()
    js = json.loads(data)
    lat = js["results"][0]["geometry"]["location"]["lat"]
    lng = js["results"][0]["geometry"]["location"]["lng"]

    return (lat, lng)

def ajustes_precio(tipo, clase, conservacion, sup_cons, sup_ter, recamaras, baños, estacionamientos, bodega, vista, \
                   roof_privado, balcon, num_nivel, num_pisos, edad):
    aj_rec = (((0.01) * (recamaras)) - 0.01) if recamaras > 1 else 0
    aj_ban = (((0.01) * (baños)) - 0.01) if baños > 1 else 0
    aj_est = .05 if ((tipo == 4) & (estacionamientos > 1)) else 0
    aJ_bod = .02 if ((tipo == 4) & (bodega == 1)) else 0
    aj_vista = .02 if vista == 1 else 0

    aj_nh1 = .05 if ((tipo == 2) & (clase == 4)) else (
        ((((1 / sup_cons) / 4) * (sup_ter - (sup_cons / num_pisos)))) if ((tipo == 2) & (clase in [5, 6])) else (
            0))

    aj_nh1 = .15 if aj_nh1 > .15 else aj_nh1

    aj_nh2 = .25 if ((tipo == 4) & (conservacion == 6) & (roof_privado == 1)) else (
        .15 if ((tipo == 4) & (conservacion == 4) & (roof_privado == 1)) else (
            0))

    aj_nh3 = .03 if ((tipo == 4) & (balcon == 1)) else 0

    aj_nh4 = .01 if ((tipo == 4) & ((num_nivel == 0) | (num_nivel == 1) | (num_nivel == 2))) else (
        .03 if ((tipo == 4) & ((num_nivel == num_pisos) | (num_nivel == (num_pisos - 1)))) else (
            0))

    aj_conser = -0.01 if ((edad >= 6) & (edad <= 10)) else (
        -0.02 if ((edad >= 11) & (edad <= 15)) else (
            -0.03 if ((edad >= 16) & (edad <= 20)) else (
                -0.04 if ((edad >= 21) & (edad <= 30)) else (
                    -0.05 if (edad > 30) else (
                        0)))))

    ajuste_total = aj_rec + aj_ban + aj_est + aJ_bod + aj_vista + aj_nh1 + aj_nh2 + aj_nh3 + aj_nh4 + aj_conser
    ajuste_total = round(ajuste_total, 2)

    return ajuste_total

def avaluo_cal(idval, latitud, longitud, cp, tipo, sup_cons, sup_ter, edad, calle, colonia, municipio, estado, \
               recamaras, baños, estacionamientos, bodega, vista, roof_privado, balcon, num_nivel, num_pisos, \
               acabados, amenidades, valor_cliente, conn):
    conservacion = 6 if int(edad) <= 2 else 4

    dic_clase_nse_cal = get_clase_nse_cal(latitud, longitud, cp, int(acabados), int(amenidades), conn)

    nse = dic_clase_nse_cal['nse']
    cal_nse = dic_clase_nse_cal['cal_nse']
    clase = dic_clase_nse_cal['clase']

    dic_m2_venta = SP_promedio(cp, tipo, clase, conservacion, sup_cons, conn)

    if dic_m2_venta == 1:
        return dic_m2_venta

    m2_avaluo = dic_m2_venta['prom_m2_venta']
    # m2_avaluo_terreno      = dic_m2_venta['prom_m2_terreno']
    # m2_avaluo_construccion = dic_m2_venta['prom_m2_construccion']

    cuenta = dic_m2_venta['cuenta']
    dif = dic_m2_venta['dif']
    id_combinacion = dic_m2_venta['IdCombinacion']
    combinacion = dic_m2_venta['combinacion']

    ajuste = ajustes_precio(tipo, clase, conservacion, sup_cons, sup_ter, recamaras, baños, estacionamientos, bodega, \
                            vista, roof_privado, balcon, num_nivel, num_pisos, edad)

    m2_avaluo = int(m2_avaluo + (m2_avaluo * ajuste))
    # m2_avaluo_terreno = int(m2_avaluo_terreno + (m2_avaluo_terreno * ajuste))
    # m2_avaluo_construccion = int(m2_avaluo_construccion + (m2_avaluo_construccion * ajuste))

    m2_minimo = int(m2_avaluo * .85)
    # m2_minimo_terreno =      int(m2_avaluo_terreno * .85)
    # m2_minimo_construccion = int(m2_avaluo_construccion * .85)

    m2_maximo = int(m2_avaluo * 1.15)
    # m2_maximo_terreno =      int(m2_avaluo_terreno * 1.15)
    # m2_maximo_construccion = int(m2_avaluo_construccion * 1.15)

    avaluo = float(sup_cons) * m2_avaluo
    avaluo = int((avaluo / 1000)) * 1000

    minimo_total = avaluo * .85
    minimo_total = int((minimo_total) / 1000) * 1000

    maximo_total = avaluo * 1.15
    maximo_total = int((maximo_total) / 1000) * 1000

    dic_cal_est = cal_estaticas(cp, m2_avaluo, conn)

    cal_transacciones = int(dic_cal_est['Score_Transacciones_SHF'])
    cal_plusvalia = int(dic_cal_est['Score_Plusvalia'])
    cal_servicios = int(dic_cal_est['Score_Servicios'])
    cal_precio = int(dic_cal_est['Score_PrecioM2'])
    cal_tm = int(dic_cal_est['Score_TimeOnMarket'])

    calificacion = round((cal_transacciones +
                          cal_plusvalia +
                          cal_servicios +
                          cal_precio +
                          cal_tm +
                          cal_nse) / 6, 2)

    calificaciones = [cal_nse, cal_transacciones, cal_plusvalia, cal_servicios, cal_precio, cal_tm, calificacion]

    calificaciones = [int((cal / 5) * 100) for cal in calificaciones]

    plusvalia = dic_cal_est['TimeOnMarket']
    tm = dic_cal_est['TimeOnMarket']

    idval_ai360 = uuid.uuid4()
    idval_ai360 = "bpb-" + str(idval_ai360)

    cdmx_current_datetime = datetime.now(pytz.timezone('America/Mexico_City'))
    cdmx_current_datetime = str(cdmx_current_datetime)[:19]

    dic_pdf = {'idval_ai360': str(idval_ai360),
               'idval': str(idval),
               'datetime': str(cdmx_current_datetime),
               'calle': str(calle),
               'colonia': str(colonia),
               'cp': str(cp),
               'municipio': str(municipio),
               'estado': str(estado),
               'latitud': latitud,
               'longitud': longitud,
               'tipo_inmueble': str(tipo),
               'edad_inmueble': str(edad),
               'conservacion': str(conservacion),
               'nse': str(nse.round(2)),
               'clase': str(clase),
               'm2_terreno': str(sup_ter),
               'm2_construccion': str(sup_cons),
               'dif': str(dif),
               'id_combinacion': str(id_combinacion),
               'combinacion': str(combinacion),
               'cuenta': str(cuenta),

               'valor_cliente': str(valor_cliente),
               'plusvalia': plusvalia,
               'time_on_market': tm,
               'estimado': int(avaluo),
               'ajuste': ajuste,
               'm2_estimado': int(m2_avaluo),
               'm2_estimado_ter': "",
               'm2_estimado_cons': "",

               'minimo_total': int(minimo_total),
               'm2_minimo': int(m2_minimo),
               'm2_minimo_ter': "",
               'm2_minimo_cons': "",

               'maximo_total': int(maximo_total),
               'm2_maximo': int(m2_maximo),
               'm2_maximo_ter': "",
               'm2_maximo_cons': "",

               'cal_nse': calificaciones[0],
               'cal_transacciones': calificaciones[1],
               'cal_plusvalia': calificaciones[2],
               'cal_servicios': calificaciones[3],
               'cal_precio': calificaciones[4],
               'cal_time_on_market': calificaciones[5],
               'calificacion_g': calificaciones[6]}

    lista_insercion = list(dic_pdf.values())
    string_sql = str(lista_insercion)
    string_sql = string_sql.replace('[', '')
    string_sql = string_sql.replace(']', '')

    insercion(string_sql, conn)

    return dict(dic_pdf)

def busqueda_com(cp, digitos, tipo, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn):
    cp = cp[:digitos]

    try:

        cursor = conn.cursor()

        sql = f"""  

                    SELECT *

                    FROM [ai360cloudprod].[dbo].[vwComparablesAPI]

                    WHERE LEFT(CODIGO_POSTAL, {str(digitos)}) = '{str(cp)}' AND
                          TIPO_INMUEBLE = '{str(tipo)}' AND
                          PRECIO_MT2 BETWEEN {minimo_m2} AND {maximo_m2} AND
                          METROS_CUADRADOS BETWEEN {int(sup_cons) - 15} AND {int(sup_cons) + 15}

               """

        cursor.execute(sql)
        df = cursor.fetchall()
        columns = cursor.description
        df = [{columns[index][0]: column for index, column in enumerate(value)} for value in df]
        df = pd.DataFrame(df)

        promedios = {'SUPERFICIE': str(int(df['METROS_CUADRADOS'].mean())),
                     'PRECIO': str(int(df['PRECIO'].mean())),
                     'PRECIO_MT2': str(int(df['PRECIO_MT2'].mean())),
                     'PROM_GLOBAL': str(int(df['PRECIO_MT2'].mean()))}

        df.replace(to_replace=[None, np.nan, 'NULL'], value="", inplace=True)

        for a, b, c in df[['LATITUD', 'LONGITUD']].itertuples():
            df.at[a, 'DISTANCIA'] = round((dist([latitud, longitud], [b, c])) / 1000, 2)

        df = df.sort_values('DISTANCIA')
        df = df.head(10)

        if len(df) >= 6:

            df['FECHA_CONSULTA'] = pd.to_datetime(df['FECHA_CONSULTA'], format='%Y/%m/%d').dt.date

            df = df.astype({'PRECIO': 'int',
                            'RECAMARAS': 'int',
                            'METROS_CUADRADOS': 'int',
                            'BANIOS': 'int',
                            'ESTACIONAMIENTOS': 'int',
                            'ANIOS': 'int',
                            'PRECIO_MT2': 'int'}, errors='ignore')

            df = df.rename(columns={'TIPO_INMUEBLE': 'type',
                                    'LATITUD': 'lat',
                                    'LONGITUD': 'lng',
                                    'CODIGO_POSTAL': 'zip_code',
                                    'PRECIO': 'price',
                                    'PRECIO_MT2': 'price_m2',
                                    'METROS_CUADRADOS': 'built_surface',
                                    'RECAMARAS': 'rooms',
                                    'BANIOS': 'bathrooms',
                                    'ESTACIONAMIENTOS': 'parking_slots',
                                    'SIMILITUD': 'homologation_factor',
                                    'DISTANCIA': 'distance',
                                    'URL_WEBSCRAPING': 'url_ad',
                                    'URL_IMAGEN': 'url_images'})

            df.replace(to_replace=[None, np.nan, 'NULL'], value="", inplace=True)
            df.drop(['cve_ent', 'cve_mun'], axis=1, inplace=True)

            df = df.astype('str', errors='ignore')

            df = df[['type', 'lat', 'lng', 'zip_code', 'price', 'price_m2', 'built_surface', 'rooms', 'bathrooms',
                     'parking_slots', 'homologation_factor', 'distance', 'url_ad', 'url_images']].copy()

            df = df.to_dict(orient='records')

            return df, promedios

        else:
            df = []

    except:

        df = []

    return df

def bd_ent_mun_cp(cp, conn):
    cursor = conn.cursor()

    sql = f"""  

                SELECT c.CodigoPostal,
                       est.cve_ent

                FROM [ai360cloudprod].[teed].[RelCPMunicipio] AS c 
                LEFT JOIN [ai360cloudprod].teed.DimMunicipio  AS mun ON c.IdMunicipio = mun.IdMunicipio
                LEFT JOIN [ai360cloudprod].teed.DimEstado     AS est ON mun.cve_ent = est.cve_ent
                WHERE CodigoPostal = {cp}


           """

    cursor.execute(sql)
    data = cursor.fetchall()
    columns = cursor.description
    data = [{columns[index][0]: column for index, column in enumerate(value)} for value in data]
    data = pd.DataFrame(data)

    data = data['cve_ent'].values[0]

    return data

def busqueda_com_est(cp, tipo, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn):
    ent = bd_ent_mun_cp(cp, conn)

    try:

        cursor = conn.cursor()

        sql = f"""  

                    SELECT *

                    FROM [ai360cloudprod].[dbo].[vwComparablesAPI]

                    WHERE cve_ent = '{str(ent)}' AND
                          TIPO_INMUEBLE = '{str(tipo)}' AND
                          PRECIO_MT2 BETWEEN {minimo_m2} AND {maximo_m2} AND
                          METROS_CUADRADOS BETWEEN {int(sup_cons) - 20} AND {int(sup_cons) + 20}

               """

        cursor.execute(sql)
        df = cursor.fetchall()
        columns = cursor.description
        df = [{columns[index][0]: column for index, column in enumerate(value)} for value in df]
        df = pd.DataFrame(df)

        promedios = {'SUPERFICIE': str(int(df['METROS_CUADRADOS'].mean())),
                     'PRECIO': str(int(df['PRECIO'].mean())),
                     'PRECIO_MT2': str(int(df['PRECIO_MT2'].mean())),
                     'PROM_GLOBAL': str(int(df['PRECIO_MT2'].mean()))}

        df.replace(to_replace=[None, np.nan, 'NULL'], value="", inplace=True)

        for a, b, c in df[['LATITUD', 'LONGITUD']].itertuples():
            df.at[a, 'DISTANCIA'] = round((dist([latitud, longitud], [b, c])) / 1000, 2)

        df = df.sort_values('DISTANCIA')
        df = df.head(10)

        if len(df) >= 5:

            df['FECHA_CONSULTA'] = pd.to_datetime(df['FECHA_CONSULTA'], format='%Y/%m/%d').dt.date

            df = df.astype({'PRECIO': 'int',
                            'RECAMARAS': 'int',
                            'METROS_CUADRADOS': 'int',
                            'BANIOS': 'int',
                            'ESTACIONAMIENTOS': 'int',
                            'ANIOS': 'int',
                            'PRECIO_MT2': 'int'}, errors='ignore')

            df = df.rename(columns={'TIPO_INMUEBLE': 'type',
                                    'LATITUD': 'lat',
                                    'LONGITUD': 'lng',
                                    'CODIGO_POSTAL': 'zip_code',
                                    'PRECIO': 'price',
                                    'PRECIO_MT2': 'price_m2',
                                    'METROS_CUADRADOS': 'built_surface',
                                    'RECAMARAS': 'rooms',
                                    'BANIOS': 'bathrooms',
                                    'ESTACIONAMIENTOS': 'parking_slots',
                                    'SIMILITUD': 'homologation_factor',
                                    'DISTANCIA': 'distance',
                                    'URL_WEBSCRAPING': 'url_ad',
                                    'URL_IMAGEN': 'url_images'})

            df.replace(to_replace=[None, np.nan, 'NULL'], value="", inplace=True)
            df.drop(['cve_ent', 'cve_mun'], axis=1, inplace=True)
            df = df.astype('str', errors='ignore')

            df = df[['type', 'lat', 'lng', 'zip_code', 'price', 'price_m2', 'built_surface', 'rooms', 'bathrooms',
                     'parking_slots', 'homologation_factor', 'distance', 'url_ad', 'url_images']]

            df = df.to_dict(orient='records')

            return df, promedios

        else:
            df = []

    except:

        df = []

    return df

def comparables_bpb(cp, tipo, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn):
    com = busqueda_com(cp, 5, tipo, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn)

    if len(com) == 0:

        com = busqueda_com(cp, 4, tipo, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn)

        if len(com) == 0:

            com = busqueda_com(cp, 3, tipo, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn)

            if len(com) == 0:

                com = busqueda_com(cp, 2, tipo, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn)

                if len(com) == 0:

                    com = busqueda_com_est(cp, tipo, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn)

                    if len(com) == 0:
                        promedios = {'SUPERFICIE': "",
                                     'PRECIO': "",
                                     'PRECIO_MT2': "",
                                     'PROM_GLOBAL': ""}

                        return ["", promedios]

    return com

def insercion(str_sql, conn):
    cursor = conn.cursor()

    sql = f"""

                INSERT INTO [ai360cloudprod].[dbo].[ApiAvaluosLog]( 
                             idval_ai360, idval, datetime, calle, colonia, cp, municipio, estado, latitud,
                             longitud, tipo_inmueble, edad_inmueble, conservacion, nse, clase, m2_terreno, m2_construccion,
                             dif, id_combinacion, combinacion, cuenta, valor_cliente, plusvalia ,time_on_market,
                             estimado, ajuste, m2_estimado, m2_estimado_ter, m2_estimado_cons, minimo_total, m2_minimo,
                             m2_minimo_ter, m2_minimo_cons, maximo_total, m2_maximo, m2_maximo_ter, m2_maximo_cons, cal_nse,
                             cal_transacciones, cal_plusvalia, cal_servicios, cal_precio, cal_time_on_market, calificacion_g)

                VALUES({str_sql})

            """

    cursor.execute(sql)
    conn.commit()

def comparables_shf(combinacion, id_combinacion, minimo_m2, maximo_m2, conn):
    combinacion = combinacion.split('_')
    id_combinacion = id_combinacion.split('_')

    cursor = conn.cursor()

    i = 0
    sql_conditions = ''
    for variable in combinacion:

        logical_operator = ' AND ' if i > 0 else ''

        if variable == 'cp2':
            sql_conditions += f"{logical_operator}left(CodigoPostal, 2) = '{id_combinacion[i]}'"

        elif variable == 'cp3':
            sql_conditions += f"{logical_operator}left(CodigoPostal, 3) = '{id_combinacion[i]}'"

        elif variable == 'cp4':
            sql_conditions += f"{logical_operator}left(CodigoPostal, 4) = '{id_combinacion[i]}'"

        elif variable == 'cp':
            sql_conditions += f"{logical_operator}left(CodigoPostal, 5) = '{id_combinacion[i]}'"

        else:
            sql_conditions += f"{logical_operator}{variable} = '{id_combinacion[i]}'"

        i += 1

    sql = f""" 

             SELECT CodigoPostal, 
                    SuperficieTerreno,
                    SuperficieConstruida,
                    tipo AS TipoInmueble,
                    conser AS Conservacion,
                    m2_terreno,
                    m2_construccion,
                    m2_venta_shf AS m2_venta,
                    ImporteValorConcluido,
                    Latitud,
                    Longitud

              FROM [ai360cloudprod].[dbo].[vwFactComparablesFinastrategySHF]

              WHERE {sql_conditions}

        """

    cursor.execute(sql)
    df = cursor.fetchall()
    columns = cursor.description
    df = [{columns[index][0]: column for index, column in enumerate(value)} for value in df]
    df = pd.DataFrame(df)

    estadisticas = df[['m2_venta']].describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]).T
    estadisticas.rename(columns={'count': 'count',
                                 'mean': 'mean',
                                 '10%': 'q10',
                                 '20%': 'q20',
                                 '30%': 'q30',
                                 '40%': 'q40',
                                 '50%': 'q50',
                                 '60%': 'q60',
                                 '70%': 'q70',
                                 '80%': 'q80',
                                 '90%': 'q90'}, inplace=True)

    estadisticas = estadisticas.astype('int', errors='ignore')
    estadisticas = estadisticas.astype('str', errors='ignore')
    estadisticas.drop(['std'], axis=1, inplace=True)

    estadisticas = estadisticas.to_dict(orient='records')
    estadisticas = estadisticas[0]

    df = df[(df['m2_venta'] >= int(minimo_m2)) & (df['m2_venta'] <= int(maximo_m2))]
    df = (df.sample(len(df))).head(6)
    df = df.to_dict(orient='records')

    shf = {'comparables': df,
           'estadisticas_ventas': estadisticas}

    return shf

def alerta(folio, diferencia):
    msg = EmailMessage()
    msg.set_content(
        f"""Alta disparidad ({diferencia}%) en el precio estimado vs el precio proporcionado por el cliente del inmueble con id {folio}""")
    msg['Subject'] = f'Revision BpB {folio}'
    msg['From'] = "alerta.ai360@gmail.com"
    msg['To'] = ["r.morales@ai360.mx", "asosa@bienparabien.com", "lgonzalez@bienparabien.com", "r.salas@ai360.mx",
                 "b.sambrano@ai360.mx"]

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login("alerta.ai360@gmail.com", "bqtakzfoknmtbhnd")
    server.send_message(msg)
    server.quit()

def validacionesCampos(dicc):
    contador = 0
    campos_validados = ""
    campo_validado = ""

    if dicc['property_details']['address'] == "" and dicc['property_details']['geolocation'] == "":
        return "Se requiere la ubcación del inmueble"

    if dicc['property_details']['id_appraisal'] == "":
        campo_validado += "id_appraisal"
        campos_validados += "id_appraisal."
        contador = contador + 1
    if dicc['property_details']['address']['zip_code'] == "":
        campo_validado += "zip_code"
        campos_validados += "zip_code."
        contador = contador + 1
    if dicc['property_details']['info']['land_surface'] == "":
        campo_validado += "land_surface"
        campos_validados += "land_surface."
        contador = contador + 1
    if dicc['property_details']['info']['built_surface'] == "":
        campo_validado += "built_surface"
        campos_validados += "built_surface."
        contador = contador + 1
    if dicc['property_details']['info']['type'] == "":
        campo_validado += "type"
        campos_validados += "type."
        contador = contador + 1
    if dicc['property_details']['info']['age'] == "":
        campo_validado += "age"
        campos_validados += "age."
        contador = contador + 1

    if contador == 1:
        respuesta_error_vacio = "El siguiente dato es necesario para realizar la consulta: " + campo_validado
        return (respuesta_error_vacio)

    if contador > 1:
        campos_validados_ordenados1 = campos_validados.replace(".", ", ", contador - 2)
        campos_validados_ordenados2 = campos_validados_ordenados1.replace(".", " y ", 1)
        respuestas_errores_vacios = "Los siguientes datos son necesarios para realizar la consulta: " + campos_validados_ordenados2

        return (respuestas_errores_vacios)

    land_surface = dicc['property_details']['info']['land_surface']
    built_surface = dicc['property_details']['info']['built_surface']

    land_surface_round = int(float(land_surface))
    built_surface_round = int(float(built_surface))

    if int(land_surface_round) <= 0 or int(built_surface_round) <= 0:
        return "Superficie Inválida"

    if int(land_surface_round) <= 30 or int(built_surface_round) <= 30:
        if int(land_surface_round) > 0 or int(built_surface_round) > 0:
            return "Superficie muy pequeña para encontrar transacciones"

    if dicc['property_details']['info']['type'] == 4:
        if int(land_surface_round) > 250 or int(built_surface_round) > 250:
            return "Superficie muy grande para departamentos"

    calle = dicc['property_details']['address']['street']
    colonia = dicc['property_details']['address']['block']
    municipio = dicc['property_details']['address']['locality']
    estado = dicc['property_details']['address']['state']
    lat = dicc['property_details']['geolocation']['lat']
    long = dicc['property_details']['geolocation']['lng']

    if (calle == "" or colonia == "" or municipio == "" or estado == "") and (lat == "" or long == ""):
        return "Se requiere la ubicación del inmueble"

    return ""

def bpbComparables(event, context):
    token = event['headers']['Authorization']
    token = token.replace('', '')
    TOKEN_BPB = os.environ.get('TOKEN_BPB')

    if token == TOKEN_BPB:
        decoded = jwt.decode(token, options={"verify_signature": False})  # works in PyJWT >= v2.0
        event = json.loads(event['body'])

        print('Request:')
        print(event)
        t1 = datetime.now()

        respuesta_validaciones = validacionesCampos(event)

        if respuesta_validaciones == "":
            check = bpb_dicc(event)

        else:
            return {
                "body": json.dumps({"Errores": {
                    "Error_1": {
                        "Descripcion": str(respuesta_validaciones),
                    }
                }
                }
                ),
                "headers": {},
                "statusCode": 400
            }
        clave = "excepcion"

        if clave in check:
            cp = check['excepcion']

            DRIVER = os.environ.get('DRIVER')
            SERVER = os.environ.get('SERVER')
            PORT = os.environ.get('PORT')
            DATABASE = os.environ.get('DATABASE')
            UID = os.environ.get('UID')
            PWD = os.environ.get('PWD')
            TDS_Version = os.environ.get('TDS_Version')

            conn = pyodbc.connect(f'DRIVER={DRIVER}; \
                                SERVER={SERVER}; \
                                PORT={PORT}; \
                                DATABASE={DATABASE}; \
                                UID={UID}; \
                                PWD={PWD}; \
                                TDS_Version={TDS_Version}')

            cursor = conn.cursor()

            sql = f"""

                SELECT m2_venta_min, 
                m2_venta_percentile_10,
                m2_venta_percentile_20,
                m2_venta_percentile_30,
                m2_venta_percentile_40,
                m2_venta_percentile_50,
                m2_venta_percentile_60,
                m2_venta_percentile_70,
                m2_venta_percentile_80,
                m2_venta_percentile_90,
                m2_venta_max,
                m2_venta_count,
                m2_venta_mean,
                Score_Total
                FROM [ai360cloudprod].[dbo].[FactAvaluoDistribucion_m2_mun] 
                WHERE CodigoPostal = {cp}

                """

            cursor.execute(sql)

            data = cursor.fetchall()
            columns = cursor.description
            data = [{columns[index][0]: column for index, column in enumerate(value)} for value in data]
            data = pd.DataFrame(data)

            dic_comb = {'m2_venta_min': data['m2_venta_min'].values[0],
                        'm2_venta_percentile_10': data['m2_venta_percentile_10'].values[0],
                        'm2_venta_percentile_20': data['m2_venta_percentile_20'].values[0],
                        'm2_venta_percentile_30': data['m2_venta_percentile_30'].values[0],
                        'm2_venta_percentile_40': data['m2_venta_percentile_40'].values[0],
                        'm2_venta_percentile_50': data['m2_venta_percentile_50'].values[0],
                        'm2_venta_percentile_60': data['m2_venta_percentile_60'].values[0],
                        'm2_venta_percentile_70': data['m2_venta_percentile_70'].values[0],
                        'm2_venta_percentile_80': data['m2_venta_percentile_80'].values[0],
                        'm2_venta_percentile_90': data['m2_venta_percentile_90'].values[0],
                        'm2_venta_max': data['m2_venta_max'].values[0],
                        'm2_venta_count': data['m2_venta_count'].values[0],
                        'm2_venta_mean': data['m2_venta_mean'].values[0],
                        'Score_Total': data['Score_Total'].values[0]
                        }
            del data

            conn.close()

            return {
                "body": json.dumps({"Errores": {
                    "Error_1": {
                        "Descripcion": "De acuerdo a los datos ingresados no se encuentran comparables",
                        "score_zona": str(dic_comb['Score_Total']),
                        "Distribucion_m2": {
                            "count": str(dic_comb['m2_venta_count']),
                            "mean": str(dic_comb['m2_venta_mean']),
                            "min": str(dic_comb['m2_venta_min']),
                            "q10": str(dic_comb['m2_venta_percentile_10']),
                            "q20": str(dic_comb['m2_venta_percentile_20']),
                            "q30": str(dic_comb['m2_venta_percentile_30']),
                            "q40": str(dic_comb['m2_venta_percentile_40']),
                            "q50": str(dic_comb['m2_venta_percentile_50']),
                            "q60": str(dic_comb['m2_venta_percentile_60']),
                            "q70": str(dic_comb['m2_venta_percentile_70']),
                            "q80": str(dic_comb['m2_venta_percentile_80']),
                            "q90": str(dic_comb['m2_venta_percentile_90']),
                            "max": str(dic_comb['m2_venta_max'])
                        }
                    }}
                }
                ),

                "headers": {},
                "statusCode": 400
            }

        if type(check['property']['requestor_details']['id_apprasial']) == str:

            print('response')
            print(check)
            # print(json.dumps(check))

            t2 = datetime.now()
            seconds = t2 - t1

            response = {
                "body": json.dumps(check)}

            return response

        else:

            check = bpb_dicc(event)

            t2 = datetime.now()
            seconds = t2 - t1

            response = {
                "body": json.dumps(check)}

    else:
        return {
            "body": json.dumps({"Errores": {
                "Error": {
                    "Descripcion": "Unauthorized",
                }
            }
            }
            ),
            "headers": {},
            "statusCode": 401
        }


def bpb_dicc(dicc: Dict[str, any]) -> Dict[str, Any]:
    DRIVER="ODBC Driver 17 for SQL Server"
    SERVER=os.environ.get("SERVER")
    PORT=os.environ.get("PORT")
    DATABASE=os.environ.get("DATABASE")
    UID=os.environ.get("UID")
    CONTR=os.environ.get("CONTR")
    TDS_Version=os.environ.get("TDS_Version")

    conn = pyodbc.connect(f'DRIVER={DRIVER}; \
                           SERVER={SERVER}; \
                           PORT={PORT}; \
                           DATABASE={DATABASE}; \
                           UID={UID}; \
                           PWD={CONTR}; \
                           TDS_Version={TDS_Version}')

    idval = str(dicc['property_details']['id_appraisal'])
    cp = str(dicc['property_details']['address']['zip_code'])
    calle = str(dicc['property_details']['address']['street']) if dicc['property_details']['address'][
                                                                      'street'] != "" else ""
    colonia = str(dicc['property_details']['address']['block']) if dicc['property_details']['address'][
                                                                       'block'] != "" else ""
    municipio = str(dicc['property_details']['address']['locality']) if dicc['property_details']['address'][
                                                                            'locality'] != "" else ""
    estado = str(dicc['property_details']['address']['state']) if dicc['property_details']['address'][
                                                                      'state'] != "" else ""
    land_surface = int(float(dicc['property_details']['info']['land_surface']))
    built_surface = int(float(dicc['property_details']['info']['built_surface']))
    sup_ter = land_surface
    sup_cons = built_surface
    tipo = int(dicc['property_details']['info']['type'])
    recamaras = int(dicc['property_details']['info']['rooms']) if dicc['property_details']['info'][
                                                                      'rooms'] != "" else ""
    baños = float(dicc['property_details']['info']['bathrooms']) if dicc['property_details']['info'][
                                                                        'bathrooms'] != "" else ""
    estacionamientos = int(dicc['property_details']['info']['parking_slots']) if dicc['property_details']['info'][
                                                                                     'parking_slots'] != "" else ""
    bodega = int(dicc['property_details']['info']['warehouse']) if dicc['property_details']['info'][
                                                                       'warehouse'] != "" else ""
    acabados = int(dicc['property_details']['info']['finishes']) if dicc['property_details']['info'][
                                                                        'finishes'] != "" else ""
    amenidades = int(dicc['property_details']['info']['amenities']) if dicc['property_details']['info'][
                                                                           'amenities'] != "" else ""
    roof_privado = int(dicc['property_details']['info']['roof_garden']) if dicc['property_details']['info'][
                                                                               'roof_garden'] != "" else ""
    balcon = int(dicc['property_details']['info']['balcony']) if dicc['property_details']['info'][
                                                                     'balcony'] != "" else ""
    vista = int(dicc['property_details']['info']['outside_view']) if dicc['property_details']['info'][
                                                                         'outside_view'] != "" else ""
    edad = int(dicc['property_details']['info']['age'])
    num_nivel = float(dicc['property_details']['info']['level']) if dicc['property_details']['info'][
                                                                        'level'] != "" else ""
    num_pisos = int(dicc['property_details']['info']['flats']) if dicc['property_details']['info'][
                                                                      'flats'] != "" else ""
    latitud = float(dicc['property_details']['geolocation']['lat']) if dicc['property_details']['geolocation'][
                                                                           'lat'] != "" else 0
    longitud = float(dicc['property_details']['geolocation']['lng']) if dicc['property_details']['geolocation'][
                                                                            'lng'] != "" else 0
    valor_cliente = int(float(dicc['property_details']['info']['estimate_value_client'])) if \
        dicc['property_details']['info']['estimate_value_client'] != "" else ""

    if (np.isnan(latitud) | np.isnan(longitud)):

        try:

            direccion = [calle, colonia, municipio, estado]
            direccion = [x for x in direccion if type(x) == str]

            latitud, longitud = coordenadas(', '.join(direccion))

        except:

            a = 'Verifique Direccion'

            return a

    cp = f'0{cp}' if len(cp) == 4 else (
        f'00{cp}' if len(cp) == 3 else (
            f'000{cp}' if len(cp) == 2 else (
                cp)))

    tipo_aux = 'Casa' if tipo in [2] else (
        'Casa en Condominio' if tipo in [3] else (
            'Departamento' if tipo in [4] else
            tipo))

    diccionario_avaluo = avaluo_cal(idval, latitud, longitud, cp, tipo, sup_cons, sup_ter, edad, calle, colonia, \
                                    municipio, estado, recamaras, baños, estacionamientos, bodega, vista, roof_privado, \
                                    balcon, num_nivel, num_pisos, acabados, amenidades, valor_cliente, conn)

    if diccionario_avaluo == 1:
        dicc_cp = {'excepcion': cp}

        return dicc_cp

    minimo_total = diccionario_avaluo['minimo_total']
    maximo_total = diccionario_avaluo['maximo_total']
    minimo_m2 = diccionario_avaluo['m2_minimo']
    maximo_m2 = diccionario_avaluo['m2_maximo']
    conservacion = diccionario_avaluo['conservacion']
    id_combinacion = diccionario_avaluo['id_combinacion']
    combinacion = diccionario_avaluo['combinacion']

    clase = int(diccionario_avaluo['clase'])
    clase = 'Interes social' if clase == 3 else (
        'Media' if clase == 4 else (
            'Residencial' if clase == 5 else (
                'Residencial Plus' if (clase == 6 or clase == 7) else (
                    ''))))

    cvesup = '1-(Hasta 50]' if sup_cons <= 50 else (
        '2-(50-75]' if sup_cons <= 75 else (
            '3-(75-100]' if sup_cons <= 100 else (
                '4-(100-150]' if sup_cons <= 150 else (
                    '5-(150-200]' if sup_cons <= 200 else (
                        '6-(200 o mas)')))))

    conservacion = 'Nuevo' if edad <= 2 else (
        'Usado' if edad > 2 else (edad))

    estadisticas_m2 = comparables_shf(id_combinacion, combinacion, minimo_m2, maximo_m2, conn)

    comparables = comparables_bpb(cp, tipo_aux, minimo_m2, maximo_m2, sup_cons, latitud, longitud, conn)

    estimado = diccionario_avaluo['estimado']

    if type(valor_cliente) == int or type(valor_cliente) == float:

        if ((estimado / valor_cliente) < .30) or ((estimado / valor_cliente) > 1.30):
            diferencia = round(((estimado / valor_cliente) - 1) * 100, 2)

            alerta(idval, diferencia)

    dicc_json_avaluo = {"property": {

        "requestor_details": {

            "id_apprasial": str(diccionario_avaluo['idval']),
            "id_apprasial_ai360": str(diccionario_avaluo['idval_ai360'])

        },

        "info": {

            "type": str(tipo),
            "lat": str(latitud),
            "lng": str(longitud),
            "street": str(calle),
            "block": str(colonia),
            "zip_code": str(cp),
            "locality": str(municipio),
            "state": str(estado),
            "age": str(edad),
            "land_surface": str(sup_ter),
            "built_surface": str(sup_cons),
            "rooms": str(recamaras),
            "bathrooms": str(baños),
            "parking_slots": str(estacionamientos)
        },

        "apprasial": {

            "conservation": str(conservacion),
            "class": str(clase),
            "time_on_market": str(diccionario_avaluo['time_on_market']),
            "updated_estimate_total": str(diccionario_avaluo['estimado']),
            "updated_estimate_m2": str(diccionario_avaluo['m2_estimado']),
            "updated_min_total": str(diccionario_avaluo['minimo_total']),
            "updated_min_m2": str(diccionario_avaluo['m2_minimo']),
            "updated_max_total": str(diccionario_avaluo['maximo_total']),
            "updated_max_m2": str(diccionario_avaluo['m2_maximo']),
            "statistics_m2": estadisticas_m2['estadisticas_ventas'],
            "similar_properties": comparables[0]
        }
    }
    }

    # insert_dicc["property"]['requestor_details'].pop('id_apprasial_ai360', None)
    # insert_dicc = {key: value for (key, value) in dicc_json_avaluo.items()}

    insert_dicc = {"id_apprasial_ai360": str(diccionario_avaluo['idval_ai360']),
                   "property": dicc_json_avaluo['property']}

    insert_dicc["property"]['requestor_details'].pop('id_apprasial_ai360', None)

    # dynamodb = boto3.resource('dynamodb',
    #                           aws_access_key_id=('AKIAU32QZGPMHT6Z2EXN'),
    #                           aws_secret_access_key=('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'),
    #                           region_name=('us-east-1'))

    # table = dynamodb.Table('avaluos_dicc_pdf')
    # table.put_item(Item = insert_dicc)

    del insert_dicc

    dicc_json_avaluo["property"]['requestor_details']['id_apprasial_ai360'] = str(diccionario_avaluo['idval_ai360'])
    del dicc_json_avaluo["property"]['info']

    json_avaluo = json.dumps(dicc_json_avaluo, indent=4)
    json_avaluo = json.loads(json_avaluo)
    conn.close()
    return json_avaluo


load_dotenv(find_dotenv())

@app.post("/api/calculo_avaluo")
async def calculo_avaluo(dicc: Dict[str, Any]= {
    "property_details": {
        "id_appraisal": 29099,
        "address": {
            "street": "Calle 2 77",
            "block": "Mozimba",
            "zip_code": "39460",
            "locality": "Acapulco",
            "state": "Guerrero"
        },
        "geolocation": {
            "lat": "20.66539836",
            "lng": "-100.40054038"
        },
        "info": {
            "type": 2,
            "land_surface": "156.92",
            "built_surface": "77.00",
            "age": 14,
            "rooms": 3,
            "bathrooms": 1,
            "parking_slots": 1,
            "warehouse": 0,
            "finishes": 0,
            "amenities": 0,
            "roof_garden": 0,
            "balcony": 0,
            "outside_view": 1,
            "level": 0,
            "flats": 2,
            "estimate_value_client": 1200000
        }
    }
}):
    json_avaluo = bpb_dicc(dicc)
    return json_avaluo





