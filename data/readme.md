# Readme

## Data

There are four datasets from three integrins and under two types of SMD simulation setups. Force-ramp SMD is the pulling simulation, in which the integrin is pulled to extend at constant velocity. Force-clamp is the simulation in which the integrin is held at a certain length at constant force.

- a5b1 Clamp ([paper](https://pubs.acs.org/doi/10.1021/acsnano.3c06253)): the integrin is clamped at four different extension levels (12, 14, 16, 18nm). These are treated as four timesteps in the DynMoCo model. 
- aVb3 Clamp ([paper](https://pubs.acs.org/doi/10.1021/acsnano.3c06253)): the integrin is clamped at four different extension levels (3, 11, 16, 18nm). These are treated as four timesteps in the DynMoCo model. 
- alpha2bbeta3 Ramp: ([paper](https://pubmed.ncbi.nlm.nih.gov/39706199/)): we extracted four equally distributed 10s segments ('0-10', '63-73', '126-136', '189-199') from the trajectory as four timesteps in the DynMoCo model. 
- alphaVbeta3 Ramp: ([paper](https://pubmed.ncbi.nlm.nih.gov/39706199/)): we extracted four equally distributed 10s segments ('0-10', '63-73', '126-136', '189-199') from the trajectory as four timesteps in the DynMoCo model. 

