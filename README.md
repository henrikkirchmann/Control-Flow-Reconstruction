# Privacy Risk Induced by Control-Flow Reconstruction from Business Process Models

# Requirements
See the "requirements.txt" file, we used Python 3.10.
# How To Use

1. Just put the XES file of the event log you want to evaluate in the folder of this project, like the `BPIC 2013 Closed Problems.xes`. 

2. Set in evaluation.py line 185 `logName` to the name of the XES file of the log we would like to evaluate.

    Example. We would like to import the "BPIC 2013 Closed Problems.xes" file:

   `logName = 'BPIC 2013 Closed Problems'` 

3. Set in evaluation.py in line 185 the number of logs `numberOfLogs` to the number of reconstructed logs each play-out strategy should generate.

    Example. When each play-out strategy should generate should reconstruct 100 logs from a process tree we type:

   `numberOfLogs = [100]`

4. Optionally: If you want a better run time, comment out the computation of the EMD, lines 255-263. 

5. Run the evaluation.py file.

