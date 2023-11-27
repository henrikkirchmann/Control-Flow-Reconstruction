'''
    This file was in the beginning part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).
    We modified PM4Pys Process Tree data structure and play-out technique to build our experiments
'''

import datetime
import random
from copy import deepcopy

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.process_tree import obj as pt_opt
from pm4py.objects.process_tree import state as pt_st
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.process_tree.utils import generic as pt_util
from pm4py.util import xes_constants as xes


class GenerationTree(ProcessTree):
    # extend the parent class to replace the __eq__ and __hash__ method

    def __init__(self, tree):
        i = 0
        self.eventFreq = 0
        self.loopCountLaplace = None
        self.loopCount = 0
        while i < len(tree.children):
            tree.children[i] = GenerationTree(tree.children[i])
            tree.children[i].parent = self
            i = i + 1
        ProcessTree.__init__(self, operator=tree.operator, parent=tree.parent, children=tree.children, label=tree.label)

    def __eq__(self, other):
        # method that is different from default one (different taus must give different ID in log generation!!!!)
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


def generateLog(pt0, no_traces=100):
    """
    Generate a log out of a process tree

    Parameters
    ------------
    pt
        Process tree
    no_traces
        Number of traces contained in the process tree

    Returns
    ------------
    log
        Trace log object
    """
    pt = deepcopy(pt0)
    # different taus must give different ID in log generation!!!!
    # so we cannot use the default process tree class
    # we use this different one!
    log = list()
    eventlog = EventLog()
    # assigns to each event an increased timestamp from 1970
    curr_timestamp = 10000000
    for i in range(no_traces):
        ex_seq, traceString = execute(pt)
        log.append(list(traceString))
        ex_seq_labels = pt_util.project_execution_sequence_to_labels(ex_seq)
        trace = Trace()
        trace.attributes[xes.DEFAULT_NAME_KEY] = str(i)
        for label in ex_seq_labels:
            event = Event()
            event[xes.DEFAULT_NAME_KEY] = label
            event[xes.DEFAULT_TIMESTAMP_KEY] = datetime.datetime.fromtimestamp(curr_timestamp)
            trace.append(event)
            curr_timestamp = curr_timestamp + 1
        eventlog.append(trace)
    return log, eventlog


def execute(pt):
    """
    Execute the process tree, returning an execution sequence

    Parameters
    -----------
    pt
        Process tree

    Returns
    -----------
    exec_sequence
        Execution sequence on the process tree
    """
    enabled, open, closed = set(), set(), set()
    enabled.add(pt)
    # populate_closed(pt.children, closed)
    execution_sequence = list()
    traceString = list()
    while len(enabled) > 0:
        execute_enabled(enabled, open, closed, execution_sequence, traceString)
    return execution_sequence, traceString


def populate_closed(nodes, closed):
    """
    Populate all closed nodes of a process tree

    Parameters
    ------------
    nodes
        Considered nodes of the process tree
    closed
        Closed nodes
    """
    closed |= set(nodes)
    for node in nodes:
        populate_closed(node.children, closed)


def execute_enabled(enabled, open, closed, execution_sequence=None, traceString=None):
    """
    Execute an enabled node of the process tree

    Parameters
    -----------
    enabled
        Enabled nodes
    open
        Open nodes
    closed
        Closed nodes
    execution_sequence
        Execution sequence

    Returns
    -----------
    execution_sequence
        Execution sequence
    """
    execution_sequence = list() if execution_sequence is None else execution_sequence
    traceString = list() if traceString is None else traceString
    # first take vertex from list where operator is not none
    vertex = None
    for vertex_enabled in enabled:
        if vertex_enabled.operator is not None:
            vertex = vertex_enabled
            break
    # choose u.a.r. next node, for parallel structure
    if vertex == None:
        for node in enabled:
            if node.label is None:
                vertex = node
        if vertex == None:
            vertex = random.sample(list(enabled), 1)[0]
    enabled.remove(vertex)
    open.add(vertex)
    execution_sequence.append((vertex, pt_st.State.OPEN))
    if len(vertex.children) > 0:
        if vertex.operator is pt_opt.Operator.LOOP:
            while len(vertex.children) < 3:
                vertex.children.append(ProcessTree(parent=vertex))
        if vertex.operator is pt_opt.Operator.LOOP:
            vertex.eventFreq -= 1
            c = vertex.children[0]
            enabled.add(c)
            execution_sequence.append((c, pt_st.State.ENABLED))
        if vertex.operator is pt_opt.Operator.SEQUENCE or vertex.operator is pt_opt.Operator.PARALLEL:
            vertex.eventFreq -= 1
            c = vertex.children[0]
            enabled.add(c)
            execution_sequence.append((c, pt_st.State.ENABLED))
        elif vertex.operator is pt_opt.Operator.XOR:
            vertex.eventFreq -= 1
            vc = list()
            for child in vertex.children:
                eventFreq = child.eventFreq
                if (eventFreq > 0):
                    vc.append(child)
                    break
            c = vc
            enabled.add(c[0])
            execution_sequence.append((c, pt_st.State.ENABLED))
    else:
        if (type(vertex) is not ProcessTree):
            vertex.eventFreq -= 1
            if vertex.label is not None:
                traceString.append(vertex.label)
        close(vertex, enabled, open, closed, execution_sequence)
    return execution_sequence, traceString


def close(vertex, enabled, open, closed, execution_sequence):
    """
    Close a given vertex of the process tree

    Parameters
    ------------
    vertex
        Vertex to be closed
    enabled
        Set of enabled nodes
    open
        Set of open nodes
    closed
        Set of closed nodes
    execution_sequence
        Execution sequence on the process tree
    """
    open.remove(vertex)
    closed.add(vertex)
    execution_sequence.append((vertex, pt_st.State.CLOSED))
    process_closed(vertex, enabled, open, closed, execution_sequence)


def process_closed(closed_node, enabled, open, closed, execution_sequence):
    """
    Process a closed node, deciding further operations

    Parameters
    -------------
    closed_node
        Node that shall be closed
    enabled
        Set of enabled nodes
    open
        Set of open nodes
    closed
        Set of closed nodes
    execution_sequence
        Execution sequence on the process tree
    """
    vertex = closed_node.parent
    if vertex is not None and vertex in open:
        if should_close(vertex, closed, closed_node):
            close(vertex, enabled, open, closed, execution_sequence)
        else:
            enable = None
            if vertex.operator is pt_opt.Operator.SEQUENCE or vertex.operator is pt_opt.Operator.INTERLEAVING or vertex.operator is pt_opt.Operator.PARALLEL:
                enable = vertex.children[vertex.children.index(closed_node) + 1]
            elif vertex.operator is pt_opt.Operator.LOOP:
                # otherwise we violate the freq of the right child of the loop operator in the simulated log
                if vertex.children[1].eventFreq == vertex.children[0].eventFreq and vertex.children[
                    0] == closed_node and vertex.children[1].eventFreq != 0:
                    enable = vertex.children[1]
                # after the right child has no freq left and is getting closed, take the left child next
                elif vertex.children[1].eventFreq == 0 and vertex.children[1] == closed_node:
                    enable = vertex.children[0]
                # after the right child has no freq left, take only left child once per loop exec
                elif vertex.children[1].eventFreq == 0 and vertex.children[0] == closed_node:
                    enable = vertex.children[2]
                # if right child got executed, always take left child
                # increment here so we can distinguish between original taus and artifial taus
                elif vertex.children[1] == closed_node:
                    # if vertex.children[1].operator is None:
                    # vertex.children[1].eventFreq -= 1
                    enable = vertex.children[0]
                # decide if we end loop or execute right child
                else:
                    vertexChildren = vertex.children[:2]
                    # wieso?
                    if vertexChildren[1].eventFreq == 0 and closed_node == vertexChildren[1]:
                        enable = vertex.children[0]
                    else:
                        probability_distribution = list()
                        ''' 
                        for child in vertexChildren:
                            probability_distribution.append(child.eventFreq)
                        '''
                        '''
                        expectedNumberOfLoops = (vertex.children[0].eventFreq / vertex.eventFreq) - 1
                        p = symbols('p')
                        probabilityOfNoRightChild = solveset(1 / p - expectedNumberOfLoops, p).args[0]
                        probabilityOfRightChild = 1 - probabilityOfNoRightChild
                        weights = [probabilityOfRightChild, probabilityOfNoRightChild]
                        '''
                        prOfNoRepeat = vertex.eventFreq / vertex.children[0].eventFreq
                        r = random.random()
                        # weights = vertex.loopdistribution
                        population = [1, 2]

                        # if we executed left child and have a choice to execute right child, make a random choice to execute right child or end loop
                        if vertex.children.index(closed_node) == 0:
                            # c = random.choices(population=population, weights=weights, k=1)
                            # enable = vertex.children[c[0]]
                            # '''
                            if r > prOfNoRepeat:
                                enable = vertex.children[1]
                            else:
                                enable = vertex.children[2]
                            # '''
                        else:
                            enable = vertex.children[0]
            if enable is not None:
                enabled.add(enable)
                execution_sequence.append((enable, pt_st.State.ENABLED))


def should_close(vertex, closed, child):
    """
    Decides if a parent vertex shall be closed based on
    the processed child

    Parameters
    ------------
    vertex
        Vertex of the process tree
    closed
        Set of closed nodes
    child
        Processed child

    Returns
    ------------
    boolean
        Boolean value (the vertex shall be closed)
    """
    if vertex.children is None:
        return True
    # child is last child of sequence that got closed, close seq vertex
    elif vertex.operator is pt_opt.Operator.LOOP:
        if child == vertex.children[2]:
            return True
        else:
            return False
    elif vertex.operator is pt_opt.Operator.SEQUENCE or vertex.operator is pt_opt.Operator.INTERLEAVING or vertex.operator is pt_opt.Operator.PARALLEL:
        return vertex.children.index(child) == len(vertex.children) - 1
    elif vertex.operator is pt_opt.Operator.XOR:
        return True
    # elif vertex.operator is pt_opt.Operator.PARALLEL:
    #   return set(vertex.children) <= closed
