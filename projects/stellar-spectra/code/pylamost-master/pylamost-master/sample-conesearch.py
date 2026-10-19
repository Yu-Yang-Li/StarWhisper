from pylamost import lamost
import io
import pandas as pd
lm=lamost(dr_version='dr10', sub_version='v2.0', is_dev=True)#init the lamost class
csv = lm.conesearch(ra=10.0004738,dec=40.9952444,radius=0.2, ismed=False, fmt='csv')
print(csv[:100])
df = pd.read_csv(io.StringIO(csv))
print(df.head())
