#!/usr/bin/python
# -*- coding: UTF-8 -*-
# packages needed: urllib.request, urllib.parse, requests, pyfits, numpy, scipy.signal
import os
import requests
import astropy.io.fits
import numpy
import scipy.signal
import matplotlib.pyplot as plt
import csv
import warnings
# 忽略urllib3的所有警告
warnings.filterwarnings('ignore', module='urllib3')

class lamost:
    def __init__(self, token=None, dr_version='dr10', sub_version='v2.0', is_dev=False):
        self.__isdev = is_dev
        self.dr_version = dr_version
        self.sub_version = sub_version
        self.email = None
        self.token = token
        if is_dev:
            self.openapi_base = 'https://www2.lamost.org/openapi'
        else:
            self.openapi_base = 'https://www.lamost.org/openapi'
        self._detect_token()

    __config=None
    __config_file_path=os.path.expanduser('~')+'/pylamost.ini'
    
    def _get_config(self, reload=False):
        if not os.path.exists(self.__config_file_path): return None
        if not reload and None!=self.__config: return self.__config
        with open(self.__config_file_path) as fh:
            self.__config={}
            for line in fh:
                if line.startswith('#'):continue
                k,v=line.split("=")
                self.__config[k.strip()]=v.strip()
        return self.__config

    def _detect_token(self):
        if self.token is not None: return True
        cf = self._get_config()
        if cf is None or 'token' not in cf.keys(): 
            print('please set your token')
            return False
        self.token=cf['token']
        return True

    def download(self, url, savedir='./', params=None, overwrite=True):
        response = requests.get(url, params=params, verify=False)
        filename = savedir + '/' + response.headers.get("Content-disposition").split('=')[1]
        if os.path.exists(filename) and not overwrite:
            return filename
        # download file to temp file
        savefile = filename+'.temp'
        # download file chunk by chunk
        with open(savefile, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # rename file
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(savefile, filename)
        return filename

    def download_catalog(self, catname, savedir='./', ismed=False, overwrite=True):
        resolution = 'mrs' if ismed else 'lrs'
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/{resolution}/catalog"
        params = {'name': catname, 'token': self.token}
        return self.download(url, savedir, params, overwrite)        

    def download_fits(self, obsid, ismed=False, savedir='./', overwrite=True):
        resolution = 'mrs' if ismed else 'lrs'
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/{resolution}/spectrum/fits"
        params = {'obsid': obsid, 'token': self.token}
        return self.download(url, savedir, params, overwrite)

    def download_png(self, obsid, savedir='./', ismed=False, overwrite=True):
        resolution = 'mrs' if ismed else 'lrs'
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/{resolution}/spectrum/png"
        params = {'obsid': obsid, 'token': self.token}
        return self.download(url, savedir, params, overwrite)

    def get_fits_csv(self, obsid, ismed=False):
        resolution = 'mrs' if ismed else 'lrs'
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/{resolution}/spectrum/fits2csv"
        params = {'obsid': obsid, 'token': self.token}
        response = requests.get(url, params=params, verify=False)
        return response.text
    
    def get_unique_id_and_related_obsids(self, obsid=None, ra=None, dec=None, radius=None):
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/get_unique_id_and_related_obsids"
        params = {'obsid': obsid, 'ra': ra, 'dec': dec, 'radius': radius, 'token': self.token}
        response = requests.get(url, params=params, verify=False)
        return response.json()

    def get_info(self, obsid, ismed=False):
        resolution = 'mrs' if ismed else 'lrs'
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/{resolution}/spectrum/info"
        params = {'obsid': obsid, 'token': self.token}
        response = requests.get(url, params=params, verify=False)
        return response.json()

    def conesearch(self, ra, dec, radius, ismed=False, fmt='votable'):
        resolution = 'mrs' if ismed else 'lrs'
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/{resolution}/voservice/conesearch"
        params = {'ra': ra, 'dec': dec, 'sr': radius, 'output.fmt': fmt, 'token': self.token}
        response = requests.get(url, params=params, verify=False)
        return response.json() if fmt == 'json' else response.text

    def ssap(self, ra, dec, radius, ismed=False, fmt='votable'):
        resolution = 'mrs' if ismed else 'lrs'
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/{resolution}/voservice/ssap"
        params = {'pos': f"{ra},{dec}", 'size': radius, 'output.fmt': fmt, 'token': self.token}
        response = requests.get(url, params=params, verify=False)
        return response.json() if fmt == 'json' else response.text

    def sql(self, sql, fmt='json'):
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/sql"
        params = {'sql': sql, 'output.fmt': fmt, 'token': self.token}
        response = requests.get(url, params=params, verify=False)
        return response.json() if fmt == 'json' else response.text

    def query_table(self, table_name, query_params):
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/query/{table_name}"
        response = requests.post(url, json=query_params, params={'token': self.token}, verify=False)
        return response.json() if query_params.get('format', 'json') == 'json' else response.text
    
    def get_query_result_count(self, sqlid, ismed=False):
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/get_query_result_count"
        params = {'sqlid': sqlid, 'token': self.token}
        response = requests.get(url, params=params, verify=False)
        return response.json()
    
    def get_query_result_by_page(self, sqlid, count, rows=10000, page=1, fmt='json'):
        if page > count:
            return []
        if page * rows > count:
            rows = count - (page - 1) * rows
        url = f"{self.openapi_base}/{self.dr_version}/{self.sub_version}/get_query_result"
        params = {'sqlid': sqlid, 'rows': rows, 'page': page, 'output.fmt': fmt, 'token': self.token}
        response = requests.get(url, params=params, verify=False)
        return response.json() if fmt == 'json' else response.text  
    
    def get_query_result(self, sqlid,fmt='json'):
        count = self.get_query_result_count(sqlid)
        pagesize=10000
        start = 1
        pageindex=1
        result=[]
        while start<count:
            arr = self.get_query_result_by_page(sqlid, count, pagesize, pageindex, fmt=fmt)
            result.extend(arr)
            start+=pagesize
            pageindex+=1
        return result
    
    def download_query_result(self, sqlid, filename, fmt='csv'):
        res=self.get_query_result(sqlid, fmt=fmt)
        f=csv.writer(open(filename,'w'))
        f.writerow(res[0].keys())  # header row
        for row in res:
            f.writerow(row.values())
    
    def read_lrs_fits(self, filename):
        hdulist = astropy.io.fits.open(filename)
        len_list=len(hdulist)
        if 1==len_list:
            head = hdulist[0].header
            scidata = hdulist[0].data
            coeff0 = head['COEFF0']
            coeff1 = head['COEFF1']
            pixel_num = head['NAXIS1'] 
            specflux = scidata[0,]
            spec_noconti = scidata[2,]
            wavelength=numpy.linspace(0,pixel_num-1,pixel_num)
            wavelength=numpy.power(10,(coeff0+wavelength*coeff1))
        elif 2==len_list:
            scidata = hdulist[1].data
            wavelength = scidata[0][2]
            specflux = scidata[0][0]
        #
        hdulist.close()
        spec_smooth_7=scipy.signal.medfilt(specflux,7)
        spec_smooth_15=scipy.signal.medfilt(specflux,15)
        return (wavelength, specflux, spec_smooth_7, spec_smooth_15)
    
    def plot_lrs_fits(self, filename):
        wavelength, specflux, spec_smooth_7, spec_smooth_15 = self.read_lrs_fits(filename)        
        plt.figure().set_size_inches(18.5, 6.5, forward=True)
        plt.plot(wavelength, specflux)
        plt.xlabel('Wavelength [Ångströms]')
        plt.ylabel('Flux')
        plt.show()
    
    def read_mrs_fits(self, filename):
        hdulist = astropy.io.fits.open(filename)
        len_list=len(hdulist)
        data={}
        for i in range(1, len_list):
            header = hdulist[i].header
            scidata = hdulist[i].data
            specflux = scidata[0][0]
            wavelength = scidata[0][2]
            data[header['EXTNAME']]={'wavelength':wavelength, 'specflux':specflux}
        hdulist.close()
        #
        return data
    
    def plot_mrs_fits(self, filename):
        data = self.read_mrs_fits(filename)        
        plt.figure().set_size_inches(18.5, 6.5, forward=True)
        for k,v in data.items():
            plt.plot(v['wavelength'], v['specflux'])
        plt.xlabel('Wavelength [Ångströms]')
        plt.ylabel('Flux')
        plt.show()