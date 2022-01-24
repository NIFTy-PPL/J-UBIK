import pandas as pd
import yaml

df = pd.read_csv('pdftable/obs.csv')

#keys = ['Obs ID', 'Target Name', 'Instrument','Grating']
with open("pdftable/keys.yaml", 'r') as cfg_file:
    cfg = yaml.safe_load(cfg_file)

p = pd.concat([df[key] for key in cfg['keys']], axis=1) 
p.to_latex('pdftable/tab.tex', index=False, longtable=True)
