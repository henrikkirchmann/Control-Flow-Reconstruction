from pm4py.objects.process_tree import obj as pt_opt

lookUpTable = {}


def eventFreqPT(GenerationTree, Log):
    i = 0
    lookUpTable = createLookUpTabel(GenerationTree)
    for trace in Log:
        activityStack = transformTraceToActivityStack(trace)
        GenerationTree, activityStack = calculateFreq(GenerationTree, activityStack, lookUpTable)
        i += 1
    # Distribution of Loops stay unchanged during the replay
    # calculateLoopDistri(GenerationTree)
    return GenerationTree


def calculateFreq(GenerationTree, activityStack, lookUpTable):
    GenerationTree.eventFreq += 1
    if GenerationTree.operator is pt_opt.Operator.XOR:
        GenerationTree, activityStack = calculateFreqOfXOR(GenerationTree, activityStack, lookUpTable)
    elif GenerationTree.operator is pt_opt.Operator.SEQUENCE:
        GenerationTree, activityStack = calculateFreqOfSequence(GenerationTree, activityStack, lookUpTable)
    elif GenerationTree.operator is pt_opt.Operator.PARALLEL:
        GenerationTree, activityStack = calculateFreqOfParallel(GenerationTree, activityStack, lookUpTable)
    elif GenerationTree.operator is pt_opt.Operator.LOOP:
        GenerationTree, activityStack = calculateFreqOfLoop(GenerationTree, activityStack, lookUpTable)
    return GenerationTree, activityStack


def calculateFreqOfXOR(GenerationTree, activityStack, lookUpTable):
    if len(activityStack) == 0:
        for child in GenerationTree.children:
            if child.operator is None and child.label is None:
                child.eventFreq += 1
                return GenerationTree, activityStack
    else:
        activity = activityStack[-1]
        # check non-tau leaf children first, they take less handling
        for child in GenerationTree.children:
            if child.label is not None and child.operator is None:
                if child.label == activity:
                    child.eventFreq += 1
                    activityStack.pop()
                    return GenerationTree, activityStack
        # check non-leaf children
        for child in GenerationTree.children:
            if child.operator is not None:
                if activityStack[-1] in lookUpTable[child.__repr__()]:
                    child, activityStack = calculateFreq(child, activityStack, lookUpTable)
                    return GenerationTree, activityStack
        # no match, choose tau, bc we know that the pt matches the trace
        for child in GenerationTree.children:
            if child.operator is None and child.label is None:
                child.eventFreq += 1
                return GenerationTree, activityStack


def calculateFreqOfSequence(GenerationTree, activityStack, lookUpTable):
    for child in GenerationTree.children:
        if child.operator is None:
            activity = activityStack[-1]
            if child.label == activity:
                child.eventFreq += 1
                activityStack.pop()
        else:
            child, activityStack = calculateFreq(child, activityStack, lookUpTable)
    return GenerationTree, activityStack


def calculateFreqOfLoop(GenerationTree, activityStack, lookUpTable):
    # check non-tau-leaf children first, they take less handling
    while (1):
        tau = 0
        # execute left child
        if GenerationTree.children[0].operator is None and GenerationTree.children[0].label is None:
            GenerationTree.children[0].eventFreq += 1
        elif GenerationTree.children[0].operator is None:
            if GenerationTree.children[0].label == activityStack[-1]:
                GenerationTree.children[0].eventFreq += 1
                activityStack.pop()
        else:
            GenerationTree.children[0], activityStack = calculateFreq(GenerationTree.children[0], activityStack,
                                                                      lookUpTable)
        if len(activityStack) == 0:
            return GenerationTree, activityStack
        # right child is leaf acivity
        if (GenerationTree.children[1].operator is None and GenerationTree.children[1].label is not None):
            if GenerationTree.children[1].label == activityStack[-1]:
                GenerationTree.children[1].eventFreq += 1
                activityStack.pop()
            else:
                return GenerationTree, activityStack
        # right child is tau, left child is leaf
        elif (GenerationTree.children[1].label is None and GenerationTree.children[1].operator is None and
              GenerationTree.children[0].operator is None):
            if GenerationTree.children[0].label == activityStack[-1]:
                GenerationTree.children[1].eventFreq += 1
            else:
                return GenerationTree, activityStack
        # right child is tau, left child is non leaf
        elif (GenerationTree.children[1].label is None and GenerationTree.children[1].operator is None and
              GenerationTree.children[0].operator is not None):
            if activityStack[-1] in lookUpTable[GenerationTree.children[0].__repr__()]:
                GenerationTree.children[1].eventFreq += 1
            else:
                return GenerationTree, activityStack
        # right child is non-leaf node
        elif GenerationTree.children[1].operator is not None:
            if activityStack[-1] in lookUpTable[GenerationTree.children[1].__repr__()]:
                GenerationTree.children[1], activityStack = calculateFreq(GenerationTree.children[1], activityStack,
                                                                          lookUpTable)
            else:
                return GenerationTree, activityStack


def calculateFreqOfParallel(GenerationTree, activityStack, lookUpTable):
    # get all possible activities (non-tau leaf nodes) from each child
    leafsOfChildren = list()
    for child in GenerationTree.children:
        leafsOfChildren.append(set())
    i = 0
    for child in GenerationTree.children:
        leafs = getLeafNodes(child)
        for leaf in leafs:
            if leaf.label is not None and leaf.operator is None:
                leafsOfChildren[i].add(leaf.label)

        i += 1
    # get for each child the possible activities of the trace in the correct ordering
    childStacks = list()
    for child in GenerationTree.children:
        childStacks.append(list())
    # get all activities from the trace until we pop a activity that does not match the activities in the nodes of the parallel structure
    # activities = copy.deepcopy(activityStack)
    activityMatchInParallel = 1
    while activityMatchInParallel == 1 and len(activityStack) > 0:
        activity = activityStack[-1]
        i = 0
        for leafSet in leafsOfChildren:
            activityMatchInChild = 0
            if activity in leafSet:
                childStacks[i].append(activity)
                activityStack.pop()
                activityMatchInChild = 1
                break
            i += 1
        if activityMatchInChild == 0:
            activityMatchInParallel = 0
    for childStack in childStacks:
        childStack.reverse()
    # all stacks empty ?
    stacksEmpty = 0
    for childStack in childStacks:
        if (len(childStack) == 0):
            stacksEmpty += 1
    if (stacksEmpty == len(childStacks)):
        return GenerationTree, activityStack

    # calculate freqs of each child with the corresponding stacks
    i = 0
    for child in GenerationTree.children:
        if child.operator is None:
            activity = childStacks[i][-1]
            if child.label == activity:
                child.eventFreq += 1
                childStacks[i].pop()
            else:
                return GenerationTree, activityStack
        # update freqs of non-leaf children
        else:
            child, weDontUseThisStack = calculateFreq(child, childStacks[i], lookUpTable)
        i += 1
    return GenerationTree, activityStack


def transformTraceToActivityStack(Trace):
    activityStack = list()
    for event in Trace._list:
        activityStack.append(event._dict.get('concept:name'))
    activityStack.reverse()
    return activityStack


def getLeafNodes(GenerationTree):
    to_visit = [GenerationTree]
    all_leafs = set()
    while len(to_visit) > 0:
        n = to_visit.pop(0)
        if n.operator is None:
            all_leafs.add(n)
        for child in n.children:
            to_visit.append(child)
    return all_leafs


def createLookUpTabel(GenerationTree):
    # get a list of all non leaf nodes
    lookUpTabel = {}
    to_visit = [GenerationTree]
    allNoneLeafNodes = list()
    while len(to_visit) > 0:
        n = to_visit.pop(0)
        if n.operator is not None:
            allNoneLeafNodes.append(n)
        for child in n.children:
            to_visit.append(child)
    for node in allNoneLeafNodes:
        to_visit = [node]
        allLeafsOfNode = set()
        while len(to_visit) > 0:
            n = to_visit.pop(0)
            if n.operator is None and n.label is not None:
                allLeafsOfNode.add(n.__repr__())
            for child in n.children:
                to_visit.append(child)
        nodestr = node.__repr__()
        lookUpTabel[nodestr] = allLeafsOfNode
    return lookUpTabel
