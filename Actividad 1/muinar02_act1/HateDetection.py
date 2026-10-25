import pathlib
import spacy
import pandas as pd
from spacy import displacy
import csv
import es_core_news_md

nlp = es_core_news_md.load()


filename = "./02Dataset_sin_procesar.csv"
lines_number = 10000
#data = ESCRIBE AQUÍ TU CÓDIGO, PARA CARGAR LOS DATOS EN UN DATAFRAME DE PANDAS

