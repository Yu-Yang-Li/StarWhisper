from pylamost import lamost
lm=lamost(dr_version='dr10', sub_version='v2.0', is_dev=True)#init the lamost class
#查询低分数据
params={
        "column_constraints": [
            {
            "column_name": "obsid",
            "constraint": "195309107",
            "operation": "equal"
            },
            {
            "column_name": "teff",
            "max": 3700,
            "min": 3500,
            "operation": "between"
            }
        ],
        "limit": 100,
        "offset": 0,
        "order": "asc",
        "pos": {
            "proximity": {
                "defaultRadius": 2,
                "proximity_nearestonly": False,
                "radecTextarea": "#ra,dec,sep\n11.455864,34.420161,2.0"
            }
        },
        "pos_group": "ra,dec",
        "showcol": [
            "obsid",
            "ra",
            "dec",
            "teff",
            "teff_err",
            "snrg",
            "snri",
            "snrr",
            "snru",
            "snrz"
        ],
        "sort": "obsid",
        "output.fmt": "json"
    }
low=lm.query_table('combined', params)
print(low)

#查询中分数据
params['column_constraints']=[]
params['pos']['proximity']['radecTextarea']='229.2822975,55.4999767,2'
params['showcol']=['obsid','mobsid','ra','dec','snr']
med=lm.query_table('med_combined', params)
print(med)
