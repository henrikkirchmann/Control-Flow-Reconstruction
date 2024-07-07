import copy
import itertools
import os
from collections import Counter as mset
from pathlib import Path
from typing import List
from collections import defaultdict

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


def get_alphabet(log: List[List[str]]) -> List[str]:
    unique_activities = set()
    for trace in log:
        for activity in trace:
            unique_activities.add(activity)
    return list(unique_activities)


def get_eventual_follows_relations_between_activities_dict(log, alphabet):
    # 0 = Never Follows
    # 1 = Sometimes Follows
    # 2 = Always Follows
    # 3 = Initialize
    # eventual_follows_relations_dict[a][b] = 2 --> b does always eventual follow a
    eventual_follows_relations_dict = defaultdict(lambda: defaultdict(int))
    for a in alphabet:
        for b in alphabet:
            eventual_follows_relations_dict[a][b] = 3

    saw_activity_before_dict = defaultdict()
    for activity in alphabet:
        saw_activity_before_dict[activity] = False

    for trace in log:
        i = 0
        trace_len = len(trace)
        for activity in trace:
            if i != trace_len:
                following_activities_set = set()
                for following_activitiy in trace[i + 1:]:
                    following_activities_set.add(following_activitiy)
                    #when we see a eventual_follows relation, change relation based on observed relation before
                    if eventual_follows_relations_dict[activity][following_activitiy] == 0:
                        eventual_follows_relations_dict[activity][following_activitiy] = 1
                    #elif eventual_follows_relations_dict[activity][following_activitiy] == 1 & 2 --> nothing todo
                    elif eventual_follows_relations_dict[activity][following_activitiy] == 3:
                        eventual_follows_relations_dict[activity][following_activitiy] = 2
                for key in eventual_follows_relations_dict[activity].keys():
                    # change the initialized value to never follows for all relations we have not seen when seeing an activity for the first time
                    if eventual_follows_relations_dict[activity][key] == 3:
                        eventual_follows_relations_dict[activity][key] = 0
                    # change all always follow relations to sometimes follow relations of relations that did not happen in this trace but are classified as always follow realtions
                    elif eventual_follows_relations_dict[activity][key] == 2:
                        if key not in following_activities_set:
                            eventual_follows_relations_dict[activity][key] == 1
            i += 1
    return {k: dict(v) for k, v in
            eventual_follows_relations_dict.items()}  # Convert inner defaultdicts to regular dicts


def compare_eventual_follows_relations(eventual_follows_relations_generated_log,
                                       eventual_follows_relations_original_log):
    # 0 = Never Follows
    # 1 = Sometimes Follows
    # 2 = Always Follows
    # 3 = Initialize
    # eventual_follows_relations_dict[a][b] = 2 --> b does always eventual follow a
    never_follows_count_original = 0
    never_follows_matching_count = 0
    sometimes_follows_count_original = 0
    sometimes_follows_matching_count = 0
    always_follows_count_original = 0
    always_follows_matching_count = 0
    for activity in eventual_follows_relations_original_log.keys():
        for activity_follows in eventual_follows_relations_original_log[activity].keys():
            original_relation = eventual_follows_relations_original_log[activity][activity_follows]
            reconstructed_relation = eventual_follows_relations_generated_log[activity][activity_follows]
            if original_relation == reconstructed_relation:
                if original_relation == 0:
                    never_follows_matching_count += 1
                elif original_relation == 1:
                    sometimes_follows_matching_count += 1
                elif original_relation == 2:
                    always_follows_matching_count += 1
            if original_relation == 0:
                never_follows_count_original += 1
            elif original_relation == 1:
                sometimes_follows_count_original += 1
            elif original_relation == 2:
                always_follows_count_original += 1

    return never_follows_count_original, never_follows_matching_count, sometimes_follows_count_original, sometimes_follows_matching_count, always_follows_count_original, always_follows_matching_count


def compare_eventual_follows_relations_fp_fn(eventual_follows_relations_generated_log,
                                             eventual_follows_relations_original_log):
    # 0 = Never Follows
    # 1 = Sometimes Follows
    # 2 = Always Follows
    # 3 = Initialize
    # eventual_follows_relations_dict[a][b] = 2 --> b does always eventual follow a
    never_follows_count_original = 0
    never_follows_matching_count = 0
    never_follows_matching_count_fp = 0
    never_follows_matching_count_fn = 0

    sometimes_follows_count_original = 0
    sometimes_follows_matching_count = 0
    sometimes_follows_matching_count_fp = 0
    sometimes_follows_matching_count_fn = 0

    always_follows_count_original = 0
    always_follows_matching_count = 0
    always_follows_matching_count_fp = 0
    always_follows_matching_count_fn = 0

    for activity in eventual_follows_relations_original_log.keys():
        for activity_follows in eventual_follows_relations_original_log[activity].keys():
            original_relation = eventual_follows_relations_original_log[activity][activity_follows]
            reconstructed_relation = eventual_follows_relations_generated_log[activity][activity_follows]

            if original_relation == reconstructed_relation:
                if original_relation == 0:
                    never_follows_matching_count += 1
                elif original_relation == 1:
                    sometimes_follows_matching_count += 1
                elif original_relation == 2:
                    always_follows_matching_count += 1

            if original_relation == 0:
                never_follows_count_original += 1
                if original_relation != reconstructed_relation:
                    never_follows_matching_count_fn += 1
                    if reconstructed_relation == 1:
                        sometimes_follows_matching_count_fp += 1
                    elif reconstructed_relation == 2:
                        always_follows_matching_count_fp += 1
            elif original_relation == 1:
                sometimes_follows_count_original += 1
                if original_relation != reconstructed_relation:
                    sometimes_follows_matching_count_fn += 1
                    if reconstructed_relation == 0:
                        never_follows_matching_count_fp += 1
                    elif reconstructed_relation == 2:
                        always_follows_matching_count_fp += 1
            elif original_relation == 2:
                always_follows_count_original += 1
                if original_relation != reconstructed_relation:
                    always_follows_matching_count_fn += 1
                    if reconstructed_relation == 0:
                        never_follows_matching_count_fp += 1
                    elif reconstructed_relation == 1:
                        sometimes_follows_matching_count_fp += 1

    #compute f1 scores
    #f1 =  2tp / (2tp + fp + fn)

    f1_never_follows = 2 * never_follows_matching_count / (2 * never_follows_matching_count + never_follows_matching_count_fp + never_follows_matching_count_fn)
    f1_sometimes_follows = 2 * sometimes_follows_matching_count / (2 * sometimes_follows_matching_count + sometimes_follows_matching_count_fp + sometimes_follows_matching_count_fn)
    f1_always_follows = 2 * always_follows_matching_count / (2 * always_follows_matching_count + always_follows_matching_count_fp + always_follows_matching_count_fn)
    f1_all = 2 * (never_follows_matching_count+sometimes_follows_matching_count+always_follows_matching_count) / (2 * (never_follows_matching_count+sometimes_follows_matching_count+always_follows_matching_count) + never_follows_matching_count_fp + sometimes_follows_matching_count_fp + always_follows_matching_count_fp + never_follows_matching_count_fn +  sometimes_follows_matching_count_fn + always_follows_matching_count_fn)


    return never_follows_count_original, never_follows_matching_count, sometimes_follows_count_original, sometimes_follows_matching_count, always_follows_count_original, always_follows_matching_count, f1_never_follows, f1_sometimes_follows, f1_always_follows, f1_all



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
eventuallyFollowsListStrategies = list()
emdListStrategies = list()
varianceCounter = 0
maxTraceLength = getLengthOfLongestTrace(log)

#for eventual follows relation evaluation
control_flow_original_log = transformLogToTraceStringList(log)
alphabet_original_log = get_alphabet(control_flow_original_log)
eventual_follows_relations_original_log = get_eventual_follows_relations_between_activities_dict(
    control_flow_original_log, alphabet_original_log)

for strategy in strategies:
    generatedLogList, generatedEventLogList = getLogs(processTreeFreq, numberOfLogs[-1], numberOfCasesInOriginalLog,
                                                      strategy,
                                                      varianceList[varianceCounter])
    originalLogList = list()
    if "D with Variance" in strategy and varianceCounter != len(varianceList) - 1:
        varianceCounter += 1
    originalLogList.append(transformLogToTraceStringList(log))

    # Trace Lengths
    ''' 
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
    '''
    # EMD
    # If you are not interested in EMD and want to improve your performance, please comment out this part.
    '''
    print("----EMD----")
    emdList = getEMD(generatedEventLogList, log)
    emdListStrategies.append(emdList)
    for i in numberOfLogs:
        print("Average EMD of the original Log and " + str(i) + " generated Logs is: " + str(getAvgOfList(emdList, i)))
        print("Maximum EMD of the original Log and " + str(i) + " generated Logs is: " + str(getMaxOfList(emdList, i)))
        print("Minimum EMD of the original Log and " + str(i) + " generated Logs is: " + str(getMinOfList(emdList, i)))
    '''

    #Eventually Follows Relation
    print("----Eventually Follows Relation----")

    eventually_follows_matching_count_list = list()
    never_follows_matching_count_list = list()
    sometimes_follows_matching_count_list = list()
    always_follows_matching_count_list = list()

    f1_always_follows_list = list()
    f1_sometimes_follows_list = list()
    f1_never_follows_list = list()
    f1_all_list = list()

    for event_log in generatedLogList:
        eventual_follows_relations_generated_log = get_eventual_follows_relations_between_activities_dict(event_log,
                                                                                                          alphabet_original_log)
        #never_follows_count_original, never_follows_matching_count, sometimes_follows_count_original, sometimes_follows_matching_count, always_follows_count_original, always_follows_matching_count = compare_eventual_follows_relations(
        #    eventual_follows_relations_generated_log, eventual_follows_relations_original_log)

        never_follows_count_original, never_follows_matching_count, sometimes_follows_count_original, sometimes_follows_matching_count, always_follows_count_original, always_follows_matching_count, f1_never_follows, f1_sometimes_follows, f1_always_follows, f1_all = compare_eventual_follows_relations_fp_fn(
            eventual_follows_relations_generated_log, eventual_follows_relations_original_log)


        never_follows_matching_count_list.append(never_follows_matching_count)
        sometimes_follows_matching_count_list.append(sometimes_follows_matching_count)
        always_follows_matching_count_list.append(always_follows_matching_count)

        f1_never_follows_list.append(f1_never_follows)
        f1_sometimes_follows_list.append(f1_sometimes_follows)
        f1_always_follows_list.append(f1_always_follows)
        f1_all_list.append(f1_all)

        eventually_follows_matching_count_list.append(
            never_follows_matching_count + sometimes_follows_matching_count + always_follows_matching_count)

    average_never_follows_matching_count = getAvgOfList(never_follows_matching_count_list,
                                                        len(never_follows_matching_count_list))
    average_sometimes_follows_matching_count = getAvgOfList(sometimes_follows_matching_count_list,
                                                            len(sometimes_follows_matching_count_list))
    average_always_follows_matching_count = getAvgOfList(always_follows_matching_count_list,
                                                         len(always_follows_matching_count_list))

    average_percentage_never_follows_matching = average_never_follows_matching_count / never_follows_count_original
    average_percentage_sometimes_follows_matching = average_sometimes_follows_matching_count / sometimes_follows_count_original
    average_percentage_always_follows_matching = average_always_follows_matching_count / always_follows_count_original

    all_average_reconstructed_follows_relation_count = average_never_follows_matching_count + average_sometimes_follows_matching_count + average_always_follows_matching_count

    average_f1_never_follows = getAvgOfList(f1_never_follows_list,len(f1_never_follows_list))
    average_f1_sometimes_follows = getAvgOfList(f1_sometimes_follows_list,len(f1_sometimes_follows_list))
    average_f1_always_follows = getAvgOfList(f1_always_follows_list,len(f1_always_follows_list))
    average_f1_all = getAvgOfList(f1_all_list,len(f1_all_list))


    print("Average Percentage of Reconstructed Eventually Follows Relations: " + str((
                                                                                                 all_average_reconstructed_follows_relation_count / (
                                                                                                     len(alphabet_original_log) * len(
                                                                                                 alphabet_original_log)))) + " (" + str(
        all_average_reconstructed_follows_relation_count) + " out of " + str(
        len(alphabet_original_log) * len(alphabet_original_log)) + ")")
    print("Average Percentage of reconstructed Eventually Always Follows Relations: " + str(
        average_percentage_never_follows_matching) + " (" + str(
        average_never_follows_matching_count) + " out of " + str(never_follows_count_original) + ")")
    print("Average Percentage of reconstructed Eventually Sometimes Follows Relations: " + str(
        average_percentage_sometimes_follows_matching) + " (" + str(
        average_sometimes_follows_matching_count) + " out of " + str(sometimes_follows_count_original) + ")")
    print("Average Percentage of reconstructed Eventually Never Follows Relations: " + str(
        average_percentage_always_follows_matching) + " (" + str(
        average_always_follows_matching_count) + " out of " + str(always_follows_count_original) + ")")

    print("-----F1-----")
    print("F1 All: " + str(average_f1_all))
    print("F1 Always Follows: " + str(average_f1_always_follows))
    print("F1 Sometimes Follows: " + str(average_f1_sometimes_follows))
    print("F1 Never Follows: " + str(average_f1_never_follows))

#########################################
# Histogram of Trace Lengths
#########################################
'''
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
'''
