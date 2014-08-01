'''A short module to quickly grab data from Stata .dta files

Notes
-----
An important feature is that data objects are cached for later
use. This cached data is used only if the original data is modified
since the last use. Otherwise, new data objects are created from the
STATA data files and dumped into Python objects or into an hdf format.
'''

import os
import cPickle
from pandas.io import pytables
import warnings
warnings.filterwarnings('ignore',category=pytables.PerformanceWarning)

def _retrieve_data(dtafile):
	'''retrieve data dictionary from STATA .dta file'''

	datafile = os.path.basename(dtafile).split('.')
	if len(datafile) != 2:
		raise ValueError('dtafile must look like "file.dta"')
	if datafile[1] != 'dta':
		raise ValueError('dtafile must have ".dta" extension')

	base    = datafile[0]
	hdf     = os.path.join('.data_cache', '{}.h5'.format(base))
	lPickle = os.path.join('.data_cache', '{}_labels.pickle'.format(base))
	vPickle = os.path.join('.data_cache', '{}_vlabels.pickle'.format(base))
	dTime   = os.path.join('.data_cache', '{}_dtime.pickle'.format(base))

	if all([os.path.isfile(d) for d in [hdf, lPickle, vPickle, dTime]]):

		if os.path.getmtime(dtafile) == cPickle.load(open(dTime, 'rb')):
			from pandas import read_hdf

			data = read_hdf(hdf, 'data')
			labels = cPickle.load(open(lPickle, 'rb'))
			vlabels = cPickle.load(open(vPickle, 'rb'))

	elif not os.path.isdir('.data_cache'):
		os.makedirs('.data_cache')

	try:
		data
	except:
		from pandas.io.stata import StataReader
		from pandas import HDFStore

		print "Data is changed or no cached data found"
		print "Creating data objects from {}".format(dtafile)

		reader = StataReader(dtafile)
		data = reader.data(convert_dates=False,convert_categoricals=False)
		labels = reader.variable_labels()
		vlabels = reader.value_labels()

		store = HDFStore(hdf)
		store['data'] = data
		cPickle.dump(labels, open(lPickle, 'wb'))
		cPickle.dump(vlabels, open(vPickle, 'wb'))
		cPickle.dump(os.path.getmtime(dtafile), open(dTime, 'wb'))

		store.close()

	return {'data':data, 'labels':labels, 'vlabels':vlabels}

def retrieve(*dtafiles):
	'''retrieve multiple data dictionaries from STATA .dta files'''

	for dtafile in dtafiles:
		yield _retrieve_data(dtafile)