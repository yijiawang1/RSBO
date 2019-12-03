# Exploration via Sample-Efficient Subgoal Design

## What do we have
### Requirements
  
  * MOE
  * Replace the corresponding files with the files in folder ```moe```, and rerun ```moe/optimal_learning/cpp/CMakeLists.txt``` and ```moe/optimal_learning/cpp/CMakeCache.txt```.

### Algorithms
  * BESD
  * EI
  * LCB
  * HyperBand

## How to run the algorithms
#### run BESD
  ```bash
  (VIRT_ENV) $ python run_misoKG_REP.py miso_gw 0 0 gw10Two1
  ```
  This is the main file for running BESD. It requests 5 inputs:
  
  1) The environment: miso_gw, miso_ky, miso_it, miso_mc
  
  2) Which_problem: 0
  
  3) Version: 0
  
  4) Replication_no: 0,1,2,...
  
  5) Problem name: 
     
     miso_gw: gw10Two2, gw20Three1.
     
     miso_ky: ky10One.
     
     miso_it: it10.
     
     miso_mc: mcf2, mcf3.
  
#### main_gpyopt.py

#### main_hb.py

#### main_ql.py
