from pylamost import lamost
import json
lm=lamost(dr_version='dr10', sub_version='v2.0', is_dev=True)#init the lamost class
sql='select * from combined limit 10'
ret=lm.sql(sql, fmt='json')
print(ret[-1])
