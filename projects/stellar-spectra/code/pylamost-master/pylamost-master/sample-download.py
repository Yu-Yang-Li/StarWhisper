#!/usr/bin/env python
# coding: utf-8


from pylamost import lamost
lm=lamost(dr_version='dr13', sub_version='v2.0', is_dev=True)
lm.download_and_plot_lrs_spectrum(obsid=577702065)

