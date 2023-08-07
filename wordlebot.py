'''
:TODO: style review and global variables
:TODO: alpha beta pruning?
:TODO: review unset expectation
:TODO: entropy vs time
:TODO: minimize expectation loss (or loss per time) for layer
:TODO: remove worst layers? Instead of trying to improve, identify worst layers and fix those...
:TODO: at any point there is some proba that we guess correctly, (word==correct thing...) account for this....
significance of entropy:
H_3^5 is expected number of layers (log(x)/log(3^5))

with perfect splitting: 3^5 bins every turn, distrib evenly, max entropy = -log(3^-5)
actually: max entropy = min(-log(3^-5), -log(corp_sz^-1) <- should have expecttion = 1
using this, we can compute entropies / expectation of perf tree:
    perf(corp_sz) = max(1, -H_3^5(corp_sz^-1))
what about expectation of tree given some guess?
    E_best = 1 + sum bin_p * perf(bin_sz) <- has max at bin_p = 1 / 3^5, 
                                        E = 1 + -(H_3^5(corp_sz^-1)+1) = perf(corp_sz)
                                        
Is there any meaningful (least?) upper bound for expectation? Would this be useful?
    - already computed expectation can act as a sort of upper bound... if another candidate has
    E_best >= curr E then never explore that node!
    
We can always find best expected subtree, and continue to update it as we learn more (basically what we arecurrently doing w/ H)
    But note that Top levels are nessecarily much more optimistic...
    
    Can we use DFS w/ expectation pruning to efficently search subtrees?
        Maybe, but min expectation is very optimistic...
        
how does guessing the actual play into expectation?
E actually is not 1+ it is (n-1)/n +... also all green / y should not be a node
  
Could use UCB1 to balance explore / exploit...   ... or UCB1 / t
    
idea: 
    if ln max(H) is bad, some edit needs to be made to tree at ln-1 or ln-i....
    note that ln-1 takes longer to search than ln
    how to identify worst node?
        how much does node contribute to expecttion: p(l2|l1)...p(ln|ln-1)E(ln)
        penalty could be p(l2|l1)...p(ln|ln-1)(E(ln)
        
note: we can also identify perfect nodes: 
    eingleton leaf nodes are perfect, E = 1
        E = 1.5 is imperfect leaf node
    parent with all leaf children is perfect: E = 2, ... E = n
        can you make a bad tree out of perfect nodes? second to last layer can be very uneven but all perfect children.
        
is general relation between corp sz and best E interesting? I'd guess some corps are much harder than others so no...
'''
import requests
import numpy as np
import time # time stuff
from functools import lru_cache

# LOAD CORPUS
file_url = "https://www-cs-faculty.stanford.edu/~knuth/sgb-words.txt"
response = requests.get(file_url)
FULL_CORPUS = response.text
FULL_CORPUS = FULL_CORPUS.split('\n')
FULL_CORPUS = np.array([x for x in FULL_CORPUS if len(x)==5])[:500]
    
def all_responses():
    '''get response labels of bins'''
    labels = [""]
    for _ in range(5):
        new_labels = []
        for l in labels:
            new_labels += [l+'X',l+'O',l+'Y']
        labels = new_labels
    return labels
    
ALL_RESPONSES = all_responses()
    
class SearchTreeNode:
    '''
    Node element of search tree.
    Tree is dynamically expanded using searh function
    '''
    def __init__(self, 
                corpus,      # set of possible answers
                parent=None, # parent node (may not need)
                ):
        self.corpus = corpus
        self.corpus_len = len(corpus)
        
        # tree attributes
        self.parent = parent
        self.leaf = False
        self.children_words = [] # word of each child node
        self.children_nodes = [] # children
        
        # candidate search attributes
        self.candidates = []
        self.candidates_min_E = []
        
        # attributes that propogate up tree when updated
        self.children_E = []       # used to find best expectation of all children
        self.updated_E = False     # set to True by get_expectation
        self.best_H = [0,self]     # node w/ max entropy candidate in child subtree
        self.updated_min_E = False # set to True by get_max_entropy_node
        
        # build node
        if self.corpus_len==0: # dont want this to happen
            pass
        elif self.corpus_len==1: # certain leaf node
            self.leaf = True
            self.updated_H = True # default best_H works for leaf
            self.E = 1
        elif self.corpus_len==2: # uncertain leaf node
            self.leaf = True
            self.updated_H = True # default best_H works for leaf
            self.E = 1.5
        else: # internal node
            t = time.time()
            self.set_candidates() # generate a candidate list
            # build child using best candidate
            t = time.time()
            self.search_best_candidate()
        
    def set_candidates(self):
        '''get candidate words to search'''
        words, min_Es = [], []
        for word in FULL_CORPUS:
            H = self.response_entropy_per_time(word) # get entropy of response bins given word
            words.append(word), Hs.append(H)
        # add best candidates to self.candidates so length is cached_candidates attribute
        words, Hs = np.array(words), np.array(Hs)
        inds = np.argsort(Hs)
        self.candidates = list(words[inds[-n:]])
        self.candidates_H = list(Hs[inds[-n:]])
        
    def bin_responses(self, word):
        '''
        Partition corpus into response bins given word as guess
        Why partition? Given query, actual, is response unique
        '''
        # right letter right place sorting
        bins = [self.corpus]
        bins_labels = ['']
        for i, char in enumerate(word):
            new_bins = []
            new_bins_labels = []
            for ws, l in zip(bins, bins_labels):
                y, x = [], []
                for w in ws:
                    if w[i]==char:
                        y.append(w)
                    else:
                        x.append(w)
                new_bins += [y, x]
                new_bins_labels += [l+'y',l+'x']
            bins = new_bins
            bins_labels = new_bins_labels
        
        # right letter wrong place sorting
        for i, char in enumerate(word):
            new_bins = []
            new_bins_labels = []
            for ws, l in zip(bins, bins_labels):
                if l[i] == 'y': # right letter right place
                    new_bins.append(ws)
                    new_bins_labels.append(l)
                    continue
                x, o = [], []
                for w in ws:
                    # number of rlwp already found in curr guess
                    o_cnt = len([c for x,c in zip(l,word) if (x=='o') and (c==char)])
                    # count unknown of char in candidate
                    w_cnt = len([c for x,c in zip(l,w) if (x!='y') and (c==char)]) 
                    # if count of char in candidate > num of char already identified
                    if w_cnt > o_cnt: 
                        o.append(w)
                    else:
                        x.append(w)
                new_bins += [x,o]
                o_l = l[:i] + 'o' + l[i+1:]
                new_bins_labels += [l, o_l]
            bins = new_bins
            bins_labels = new_bins_labels
            
        return [b for b in bins if len(b)>0]
        
    def response_entropy_per_time(self, word):
        ''' return entropy per search time '''
        return self.response_entropy(word) / np.log(self.corpus_len)
        
    def response_entropy(self, word):
        '''
        Sort corpus into response bins given word and compute entropy of distribution
        '''
        bins = self.bin_responses(word)
                    
        ps = [len(x) / self.corpus_len for x in bins]
        H = sum([-p*np.log(p) for p in ps if p > 0])
        return H
        
    def min_expectation(self, word):
        '''
        Assuming optimal children, find minimum expected number of guesses from current node w/
        guess word
        '''
        bin_szs = [len(x) for x in self.bin_responses(word)]
        E = 1
        for b in bin_szs:
            perf_child = max(1, -np.log(1/b) / np.log(3**5))
            E += (b / self.corp_sz) * perf_child
        return E
        
    def unset_parents_E(self):
        '''
        Unset parents known best entropy value. Propogates all the way up tree
        :TODO: Only needs to propogate to nodes where calling child has min expectation.
        '''
        if self.parent is not None:
            self.parent.updated_E = False
            self.parent.unset_parents_E()
            
    def get_expectation(self):
        '''
        Get min expectation over all subtrees
        '''
        if self.leaf:
            return self.E
        if self.updated_E is False:
            children_E = []
            for c in self.children_nodes:
                E = 1
                for node in c:
                    E += node.get_expectation() * node.corpus_len / self.corpus_len
                children_E.append(E)
            self.children_E = children_E
            self.updated_E = True
        return 1 + min(self.children_E)
        
    def unset_parents_H(self):
        '''
        Unset parents known best entropy value. Propogates all the way up tree.
        '''
        if self.parent is not None:
            self.parent.updated_H = False
            self.parent.unset_parents_H()
            
    def get_max_entropy_node(self):
        '''
        Find node with highest entropy in subtree
        '''
        if self.updated_H is False:
            self.best_H = [max(self.candidates_H), self]
            for c in self.children_nodes:
                for node in c:
                    H = node.get_max_entropy_node()
                    if H[0] > self.best_H[0]:
                        self.best_H = H
            self.updated_H = True
        return self.best_H
        
    def search_best_candidate(self):
        '''
        Create new subtree at current node by searching the best candidate
        Update expectation and candidate list
        '''
        # get word
        i = np.argmax(self.candidates_H)
        word, _ = self.candidates.pop(i), self.candidates_H.pop(i)
        
        # build child subtree
        # get corpus for all possible responses
        response_bins = self.bin_responses(word)
        response_nodes = []
        for b in response_bins:
            if len(b)==0:
                continue
            child_node = SearchTreeNode(b, parent=self, cached_candidates=self.cached_candidates)
            response_nodes.append(child_node)

        # add child
        self.children_words.append(word)
        self.children_nodes.append(response_nodes)
        
        # entropy tree
        if len(self.candidates)==0:
            self.set_candidates()
        self.get_max_entropy_node() # sets H on self and all children
        self.unset_parents_H() # unsets H on all parents
        
        # expectation tree
        self.get_expectation() # sets E on self and all children
        self.unset_parents_E() # unsets E on all parents
            
    def search(self):
        '''
        Search next best candidate in subtree
        '''
        node = self.get_max_entropy_node()[1]
        node.search_best_candidate()
        
    def rank_guesses(self):
        '''
        Return sorted list of children and respective expectations
        '''
        self.get_max_entropy_node() # set self.children_E attribute
        inds = np.argsort(self.children_E)[::-1]
        guesses = []
        for i in inds:
            guesses.append((self.children_words[i], 1 + self.children_E[i]))
        return guesses
        
    
if __name__=='__main__':

    root = SearchTreeNode(FULL_CORPUS)
    print('created')
    E = 0
    for i in range(100000):
        root.search()
        guesses = root.rank_guesses()
        new_E = guesses[0][1]
        if E != new_E:
            print(i)
            print(guesses)
            E = new_E
    
    
    
    
    
    
    
    
    
    
    

