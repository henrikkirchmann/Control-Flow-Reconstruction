import copy
import itertools
import os
from collections import Counter as mset
from pathlib import Path

import pandas as pd
import pm4py
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.variants.log import get as variants_module

from frequencyAnnotationOfProcessTree import eventFreqPT
from strategyA import generateLog as generateLogUniform
from strategyB import generateLog as generateLogStaticDistribution
from strategyC import generateLog as generateLogDynamicDistribution
from strategyDwithVarianceD import GenerationTree
from strategyDwithVarianceD import generateLog as generateLogFreq
from strategyInQuantification import generateLog as generateLogMA


def getLengthOfLongestTrace(log):
    trace_lengths = [len(trace) for trace in log]
    return max(trace_lengths)


def transformLogToTraceStringList(log):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    for trace in log:
        for event in trace._list:
            log_list[i].append(event._dict.get('concept:name'))
        i += 1
    return log_list


def transformLogInStringList(log):
    stringList = list()
    for trace in log:
        traceString = ""
        for event in trace:
            traceString += (" " + event)
        stringList.append(traceString)
    return stringList


def getLogs(processTree, numberOfLogsToGenerate, numberOfCasesInOriginalLog, strategy, variance):
    # list of all Eventlog() 's generated
    generatedEventLogList = list()
    # list of Logs that have their traces as strings
    generatedLogList = list()
    print('######################')
    print("Strategy " + strategy)
    print('######################')
    print("Start generating play-outs")
    for i in range(numberOfLogsToGenerate):
        processTreeCopy = copy.deepcopy(processTree)
        if strategy == "A":
            log, eventlog = generateLogUniform(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == "B":
            log, eventlog = generateLogStaticDistribution(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == "C":
            log, eventlog = generateLogDynamicDistribution(processTreeCopy, numberOfCasesInOriginalLog)
        elif strategy == ("D with Variance " + str(variance)):
            log, eventlog = generateLogFreq(processTreeCopy, numberOfCasesInOriginalLog, variance)
        elif strategy == citationMA:
            log, eventlog = generateLogMA(processTreeCopy, numberOfCasesInOriginalLog)
        generatedLogList.append(log)
        generatedEventLogList.append(eventlog)
        print("All play-outs are generated")

    return generatedLogList, generatedEventLogList


def getMinOfList(logList, numberOfLogsToEvaluate):
    return min(logList[:numberOfLogsToEvaluate])


def getMaxOfList(logList, numberOfLogsToEvaluate):
    return max(logList[:numberOfLogsToEvaluate])


def getAvgOfList(logList, numberOfLogsToEvaluate):
    numberOfTraceVariants = 0
    for i in range(numberOfLogsToEvaluate):
        numberOfTraceVariants += logList[i]
    avgNumberOfTraceVariants = numberOfTraceVariants / numberOfLogsToEvaluate
    return avgNumberOfTraceVariants


def getTraceLengthsList(logList):
    logTraceLengths = list()
    for log in logList:
        traceLengths = list()
        for trace in log:
            traceLengths.append(len(trace))
        logTraceLengths.append(traceLengths)
    return logTraceLengths


def getMultiSetIntersection(generatedLogList, originalLogList):
    multiSetIntersectionSizeList = list()
    for generatedLog in generatedLogList:
        intersection = mset(transformLogInStringList(originalLogList)) & mset(transformLogInStringList(generatedLog))
        multiSetIntersectionSizeList.append(len(list(intersection.elements())))
    return multiSetIntersectionSizeList


def getEMD(generatedEventLogList, originalEventLog):
    emdList = list()
    for log in generatedEventLogList:
        originalLogLanguage = variants_module.get_language(originalEventLog)
        generatedLogLanguage = variants_module.get_language(log)
        emd = emd_evaluator.apply(generatedLogLanguage, originalLogLanguage)
        emdList.append(emd)
    return emdList


def transfromTraceLengthsToDataframesHistograms(originalTL: list[list], generatedTLs: list[list], numberOfLogs: list,
                                                strategies: list, logname):
    originalTLscaledToNumberOfLogs = []
    weights = []
    weights.extend([1 / numberOfLogs[-1]] * len(strategies) * len(originalTL[0]) * numberOfLogs[-1] * 2)

    for i in range(numberOfLogs[-1]):
        originalTLscaledToNumberOfLogs.extend(originalTL[0])
    traceLengths = []
    for logs in generatedTLs:
        traceLengths.extend(originalTLscaledToNumberOfLogs)
        traceLengths.extend(logs)
    st = []
    for strategy in strategies:
        if strategy == "Quantifying the Re-identification\nRisk in Published Process Models":
            strategy = "Quantifying the\nRe-identification Risk in Published\nProcess Models"
        if strategy == "D with Variance 0.5":
            strategy = "D with\nVariance 0.5"
        if strategy == "D with Variance 1":
            strategy = "D with\nVariance 1"
        if strategy == "D with Variance 3":
            strategy = "D with\nVariance 3"
        if strategy == "D with Variance 5":
            strategy = "D with\nVariance 5"
        st.extend(['Play-Out Strategy ' + strategy] * (len(originalTL[0]) * numberOfLogs[-1] * 2))
    originalLogOrNotList = []
    for _ in strategies:
        ogORnotList = []
        ogORnotList.extend(
            ['Trace Length Distribution of the Original ' + logname + ' Log'] * (len(originalTL[0])) * numberOfLogs[-1])
        ogORnotList.extend(
            ['Average Trace Length Distribution of ' + str(numberOfLogs[-1]) + ' Play-Outs'] * (len(originalTL[0])) *
            numberOfLogs[-1])
        originalLogOrNotList.extend(ogORnotList)
    tupels = list(zip(traceLengths, st, originalLogOrNotList, weights))
    return pd.DataFrame(tupels, columns=['Trace Length', 'Play-Out Strategy', 'Trace Length Distribution', 'Weights'])


def getAvgHistoOverlap(traceLengthsOG: [[]], traceLengthsListStrategies: [[]], numberOfLogs: []):
    lengthsOG = list(set(traceLengthsOG[0]))
    avgHistoOverlap = []
    for strategyList in traceLengthsListStrategies:
        count = 0
        for length in lengthsOG:
            stCount = strategyList.count(length)
            ogCount = traceLengthsOG[0].count(length) * numberOfLogs[-1]
            diff = stCount - ogCount
            if diff <= 0:
                count += stCount
            else:
                count += ogCount
        avgHistoOverlap.append(count / (len(traceLengthsOG[0]) * numberOfLogs[-1]))
    return avgHistoOverlap


# BPIC 2015 Municipality 1
# BPIC 2017
# Sepsis Cases
# BPIC 2013 Closed Problems

###########################################
logName = "BPIC 2013 Closed Problems"
numberOfLogs = [1]
###########################################

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

print(logName)
print("Number of Play-Outs:" + str(numberOfLogs[0]))
citationMA = "Quantifying the Re-identification\nRisk in Published Process Models"
varianceList = [0.5, 1, 3, 5]  # set variance for play-out strategy D
log = xes_importer.apply(logName + '.xes')

processTree = pm4py.discover_process_tree_inductive(log)
print("Process Tree is discovered")
# Transform the Process Tree Datastructures to a Process Tree Datastructures with Weights
processTreeFreq = GenerationTree(processTree)
# Replay the original Log on the Process Tree and annotate the Process Tree accordingly
processTreeFreq = eventFreqPT(processTreeFreq, log)
print("Process Tree is annotated with frequency information")

numberOfCasesInOriginalLog = processTreeFreq.eventFreq

strategies = ["A", "B", "C"]
for variance in varianceList:
    strategies.append("D with Variance " + str(variance))
strategies.append(
    citationMA)
numberOfTraceVariantListStrategies = list()
numberOfTraceLengthsListStrategies = list()
multiSetIntersectionSizeListStrategies = list()
emdListStrategies = list()
varianceCounter = 0
maxTraceLength = getLengthOfLongestTrace(log)

for strategy in strategies:
    generatedLogList, generatedEventLogList = getLogs(processTreeFreq, numberOfLogs[-1], numberOfCasesInOriginalLog,
                                                      strategy,
                                                      varianceList[varianceCounter])
    originalLogList = list()
    if "D with Variance" in strategy and varianceCounter != len(varianceList) - 1:
        varianceCounter += 1
    originalLogList.append(transformLogToTraceStringList(log))

    # Trace Lengths

    numberOfTraceLengthsList = getTraceLengthsList(generatedLogList)
    numberOfTraceLengthsListStrategies.append(list(itertools.chain.from_iterable(numberOfTraceLengthsList)))
    numberOfTraceLengthsListOriginalLog = getTraceLengthsList(originalLogList)
    print("----NHI Size----")
    nhi = str(
        getAvgHistoOverlap(numberOfTraceLengthsListOriginalLog, [numberOfTraceLengthsListStrategies[-1]], numberOfLogs)[
            0])
    print("NHI of the original Log and " + str(numberOfLogs[-1]) + " generated Logs is: " + str(nhi))

    # Intersection with multi sets
    print("----Multi Set Intersection----")
    multiSetIntersectionSizeList = getMultiSetIntersection(generatedLogList, originalLogList[0])
    multiSetIntersectionSizeListStrategies.append(multiSetIntersectionSizeList)
    for i in numberOfLogs:
        print("Average Size of the Multi Set Intersection with the original Log in " + str(
            i) + " generated Logs is: " + str(getAvgOfList(multiSetIntersectionSizeList, i)))
        print("Maximum Size of the Multi Set Intersection with the original Log in " + str(
            i) + " generated Logs is: " + str(getMaxOfList(multiSetIntersectionSizeList, i)))
        print("Minimum Size of the Multi Set Intersection with the original Log in " + str(
            i) + " generated Logs is: " + str(getMinOfList(multiSetIntersectionSizeList, i)))

    # EMD
    # If you are not interested in EMD and want to improve your performance, please comment out this part.
    #'''
    print("----EMD----")
    emdList = getEMD(generatedEventLogList, log)
    emdListStrategies.append(emdList)
    for i in numberOfLogs:
        print("Average EMD of the original Log and " + str(i) + " generated Logs is: " + str(getAvgOfList(emdList, i)))
        print("Maximum EMD of the original Log and " + str(i) + " generated Logs is: " + str(getMaxOfList(emdList, i)))
        print("Minimum EMD of the original Log and " + str(i) + " generated Logs is: " + str(getMinOfList(emdList, i)))
    #'''

#########################################
# Histogram of Trace Lengths
#########################################

# x-axis cut off for better visulasation
if logName == 'BPIC 2017':
    maxLength = 100
elif logName == 'Sepsis Cases':
    maxLength = 60
elif logName == 'BPIC 2015 Municipality 1':
    maxLength = 90
elif logName == 'BPIC 2013 Closed Problems':
    maxLength = 30
else:
    maxLength = maxTraceLength + 20

pdList = transfromTraceLengthsToDataframesHistograms(numberOfTraceLengthsListOriginalLog,
                                                     numberOfTraceLengthsListStrategies,
                                                     numberOfLogs, strategies, logName)
avgHOL = getAvgHistoOverlap(numberOfTraceLengthsListOriginalLog, numberOfTraceLengthsListStrategies, numberOfLogs)

rc('font', **{'family': 'serif', 'size': 20})

ax = sns.displot(
    pdList, x="Trace Length", col="Play-Out Strategy", hue='Trace Length Distribution', weights='Weights',
    binwidth=1, col_wrap=4,
    aspect=1 * 0.5, facet_kws=dict(margin_titles=True, despine=False), kind="hist"
)

sns.move_legend(
    ax, "center", bbox_to_anchor=(.5, .95), shadow=True,
    ncol=2, title=None, frameon=True, fancybox=True,
)

ax1 = ax.axes[0]
ax1.text(1.2, 1.35, " 123", horizontalalignment='right',
         verticalalignment='top',
         transform=ax1.transAxes)

ax.set_titles('{col_name}')
ax.set(xlim=(0, maxLength))
plt.tight_layout()
Path("pdf/" + logName + "/Histogram").mkdir(parents=True, exist_ok=True)
Path("png/" + logName + "/Histogram").mkdir(parents=True, exist_ok=True)
plt.savefig("pdf/" + logName + "/Histogram/" + str(numberOfLogs[-1]) + ".pdf", format="pdf",
            transparent=True)
plt.savefig("png/" + logName + "/Histogram/" + str(numberOfLogs[-1]) + ".png", format="png", dpi=300,
            transparent=True)
plt.show()
