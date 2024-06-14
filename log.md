# Hypothesis

## Data
### Data not a powerlaw after exposure correction.
- Test 1: Look at exp corrected data with and without effective area
  looks normal, shouldn't be a problem.
  
- Test 2: Fit with 2 Bins
  DELAY or not neccessary
  
### Does Map work
- Testing with config ve_1.yaml (get Chi² of 1 and below)
 on this git checkout ff8848e8c852b37d6d5654e2660f91ddb11ade9c
 - on small Fov (vincent)
 - on large Fov (margret)

### Does VI work?
- NO, but why?
- 1: geoVI gets lost? - MGVI only -> look at minisanity then (Margret)
- 1: the model cannot fit the data, because some DOF are missing or very unlikely, it tries to go somewhere and gradients point in the wrong direction (WAIT) - Switch on DEVS?


## Mock
- works with Map (Good Chi^² but no comp separation, but this is seen to be normal)

- VI fails
- 1: check configs (priors are off)- is the same

- 2: let's look at the minisanity (Not Sampling enough) - We try with lower xtol/absdelta. (Margret)

- 3: let's look at the diagnostics (what is going wrong with chi2, describe where the bigest problem is) (Matteo)
- 4: Frank Hypothesis: too may DOF / Deg. switching of DEVs 
- 5: too broad priors: check with smaller stds (Margret)

## MISC
