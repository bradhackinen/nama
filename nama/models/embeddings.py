import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import copy
from collections import Counter
import torch
from zipfile import ZipFile
import pickle
from io import BytesIO

from ..match_groups import MatchGroups


class Embeddings(torch.nn.Module):
    """
    Stores embeddings for a fixed array of strings and provides methods for
    clustering the strings to create MatchGroups objects according to different
    algorithms.
    """
    def __init__(self,strings,V,score_model,weighting_function,counts,device='cpu'):
        super().__init__()

        self.strings = np.array(list(strings))
        self.string_map = {s:i for i,s in enumerate(strings)}
        self.V = V
        self.counts = counts
        self.w = weighting_function(counts)
        self.score_model = score_model
        self.weighting_function = weighting_function
        self.device = device

        self.to(device)

    def __repr__(self):
        return f'<nama.Embeddings containing {self.V.shape[1]}-d vectors for {len(self)} strings'

    def to(self,device):
        super().to(device)
        self.V = self.V.to(device)
        self.counts = self.counts.to(device)
        self.w = self.w.to(device)
        self.score_model.to(device)
        self.device = device

    def save(self,f):
        """
        Save embeddings in a simple custom zipped archive format (torch.save
        works too, but it requires huge amounts of memory to serialize large
        embeddings objects).
        """
        with ZipFile(f,'w') as zip:

            # Write score model
            zip.writestr('score_model.pkl',pickle.dumps(self.score_model))

            # Write score model
            zip.writestr('weighting_function.pkl',pickle.dumps(self.weighting_function))

            # Write string info
            strings_df = pd.DataFrame().assign(
                                        string=self.strings,
                                        count=self.counts.to('cpu').numpy())
            zip.writestr('strings.csv',strings_df.to_csv(index=False))

            # Write embedding vectors
            byte_io = BytesIO()
            np.save(byte_io,self.V.to('cpu').numpy(),allow_pickle=False)
            zip.writestr('V.npy',byte_io.getvalue())

    def __getitem__(self,arg):
        """
        Slice a Match Groups object
        """
        if isinstance(arg,slice):
            i = arg
        elif isinstance(arg, MatchGroups):
            return self[arg.strings()]
        elif hasattr(arg,'__iter__'):
            # Return a subset of the embeddings and their weights
            string_map = self.string_map
            i = [string_map[s] for s in arg]

            if i == list(range(len(self))):
                # Just selecting the whole match groups object - no need to slice the embedding
                return copy(self)
        else:
            raise ValueError(f'Unknown slice input type ({type(input)}). Can only slice Embedding with a slice, match group, or iterable.')

        new = copy(self)
        new.strings = self.strings[i]
        new.V = self.V[i]
        new.counts = self.counts[i]
        new.w = self.w[i]
        new.string_map = {s:i for i,s in enumerate(new.strings)}

        return new

    def embed(self,grouping):
        """
        Construct updated Embeddings with counts from the input MatchGroups
        """
        new = self[grouping]
        new.counts = torch.tensor([grouping.counts[s] for s in new.strings],device=self.device)
        new.w = new.weighting_function(new.counts)

        return new

    def __len__(self):
        return len(self.strings)

    def _group_to_ids(self,grouping):
        group_id_map = {g:i for i,g in enumerate(grouping.groups.keys())}
        group_ids = torch.tensor([group_id_map[grouping[s]] for s in self.strings]).to(self.device)
        return group_ids

    def _ids_to_group(self,group_ids):
        if isinstance(group_ids,torch.Tensor):
            group_ids = group_ids.to('cpu').numpy()

        strings = self.strings
        counts = self.counts.to('cpu').numpy()

        # Sort by group and string count
        g_sort = np.lexsort((counts,group_ids))
        group_ids = group_ids[g_sort]
        strings = strings[g_sort]
        counts = counts[g_sort]

        # Identify group boundaries and split locations
        split_locs = np.nonzero(group_ids[1:] != group_ids[:-1])[0] + 1

        # Get grouped strings as separate arrays
        groups = np.split(strings,split_locs)

        # Build the groupings
        grouping = MatchGroups()
        grouping.counts = Counter({s:int(c) for s,c in zip(strings,counts)})
        grouping.labels = {s:g[-1] for g in groups for s in g}
        grouping.groups = {g[-1]:list(g) for g in groups}

        return grouping

    @torch.no_grad()
    def _fast_unite_similar(self,group_ids,threshold=0.5,progress_bar=True,batch_size=64):

        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)

        for batch_start in tqdm(range(0,len(self),batch_size),
                                    delay=1,desc='Predicting matches',disable=not progress_bar):

            i_slice = slice(batch_start,batch_start+batch_size)
            j_slice = slice(batch_start+1,None)

            g_i = group_ids[i_slice]
            g_j = group_ids[j_slice]

            # Find j's with jaccard > threshold ("matches")
            batch_matched = (V[i_slice]@V[j_slice].T >= cos_threshold) \
                            * (g_i[:,None] != g_j[None,:])

            for k,matched in enumerate(batch_matched):
                if matched.any():
                    # Get the group ids of the matched j's
                    matched_groups = g_j[matched]

                    # Identify all embeddings in these groups
                    ids_to_group = torch.isin(group_ids,matched_groups)

                    # Assign all matched embeddings to the same group
                    group_ids[ids_to_group] = g_i[k].clone()

        return self._ids_to_group(group_ids)

    @torch.no_grad()
    def unite_similar(self,
                threshold=0.5,
                group_threshold=None,
                always_match=None,
                never_match=None,
                batch_size=64,
                progress_bar=True,
                always_never_conflicts='warn',
                return_united=False):

        """
        Unite embedding strings according to predicted pairwise similarity.

        - "theshold" sets the minimimum match similarity required to unite two strings.
            - Note that strings with similarity<threshold can end up matched if they are
              linked by a chain of sufficiently similar strings (matching is transitive).
              "group_threshold" can be used to add an additional constraing on the minimum
              similarity within each group.
        - "group_threshold" sets the minimum similarity required within a single group.
        - "always_match" takes any argument that can be used to unite strings. These 
            strings will always be matched.
        - "never_match" takes a set, or a list of sets, where each set indicates two or
            more strings that should never be united with each other (these strings may 
            still be united with other strings).
        - "always_never_conflicts" determines how to handle conflicts between 
            "always_match" and "never_match":
            - always_never_conflicts="warn": Check for conflicts and print a warning
                if any are found (default)
            - always_never_conflicts="raise": Check for conflicts and raise an error
                if any are found
            - always_never_conflicts="ignore": Do not check for conflicts ("always_match"
              will take precedence)

        If "group_threshold" or "never_match" arguments are supplied, strings pairs are
        united in order of similarity. Highest similarity strings are matched first, and 
        before each time a new pair of strings is united, the function checks if this will
        result in grouping any two strings with similarity<group_threshold. If so, this
        pair is skipped. This version of the algorithm requires more memory and processing
        time, but guaruntees deterministic output that is consistent with the constraints.
            
        returns: MatchGroups object
        """
        if group_threshold and group_threshold < threshold:
            raise ValueError('group_threshold must be greater than or equal to threshold')

        group_ids = torch.arange(len(self)).to(self.device)
        
        if always_match is not None:
            always_grouping = (MatchGroups(self.strings)
                            .unite(always_match))
            always_match_labels = always_grouping.labels


        # Use a simpler, faster prediction algorithm if possible
        if not (return_united or group_threshold or (never_match is not None)):
            if always_match is not None:
                group_ids = self._group_to_ids(always_grouping)

            return self._fast_unite_similar(
                        group_ids=group_ids,
                        threshold=threshold,
                        batch_size=batch_size,
                        progress_bar=progress_bar)

        if never_match is not None:
            # Ensure never_match is a nested list
            if all(isinstance(s,str) for s in never_match):
                never_match = [never_match]

            if always_match is not None:

                assert always_never_conflicts in ['raise','warn','ignore']
                
                if always_never_conflicts != 'ignore':

                    # Find conflicts between never_match and always_match groups
                    conflicts = []
                    for i,g in enumerate(never_match):
                        g = sorted(list(g))
                        g_labels = [always_match_labels.get(s,s) for s in g]
                        if len(set(g_labels)) < len(g):
                            df = (pd.DataFrame()
                                  .assign(
                                    string=g,
                                    never_match_group=i,
                                    always_match_group=g_labels
                                    ))
                            conflicts.append(df)

                    if conflicts:
                        conflicts_df = pd.concat(conflicts)

                        if always_never_conflicts == 'warn':
                            print(f'Warning: The following never_match groups are in conflict with always_match groups:\n{conflicts_df}')
                            print('Conflicted never_match relationships will be ignored')
                        else:
                            raise ValueError(f'The following never_match groups are in conflict with always_match groups\n{conflicts_df}')
                                

                # If always_match, collapse to group labels that should not match
                # Note: Implicitly letting always_match over-ride never_match here
                never_match = [{always_match_labels[s] for s in g if s in always_match_labels} for g in never_match]
            
            else:
                # Otherwise just use the strings themselves as labels
                never_match = [set(s) for s in never_match]

        # Convert thresholds from scores to raw cosine distances
        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)
        if group_threshold is not None:
            separate_cos = self.score_model.score_to_cos(group_threshold)

        # First collect all pairs to match (can be memory intensive!)
        matches = []
        cos_scores = []
        for batch_start in tqdm(range(0,len(self),batch_size),
                                    desc='Scoring pairs',
                                    delay=1,disable=not progress_bar):

            i_slice = slice(batch_start,batch_start+batch_size)
            j_slice = slice(batch_start+1,None)

            # Find j's with jaccard > threshold ("matches")
            batch_cos = V[i_slice]@V[j_slice].T

            # Search upper diagonal entries only
            # (note j_slice starting index is offset by one)
            batch_cos = torch.triu(batch_cos)

            bi,bj = torch.nonzero(batch_cos >= cos_threshold,as_tuple=True)

            if len(bi):
                # Convert batch index locations to global index locations
                i = bi + batch_start
                j = bj + batch_start + 1

                cos = batch_cos[bi,bj]

                # Can skip strings that are already matched in the base grouping
                unmatched = group_ids[i] != group_ids[j]
                i = i[unmatched]
                j = j[unmatched]
                cos = cos[unmatched]

                if len(i):
                    batch_matches = torch.hstack([i[:,None],j[:,None]])

                    matches.append(batch_matches.to('cpu').numpy())
                    cos_scores.append(cos.to('cpu').numpy())

        # Unite potential match pairs in priority order, while respecting
        # the group_threshold and never_match arguments
        united = []
        if matches:
            matches = np.vstack(matches)
            cos_scores = np.hstack(cos_scores).T

            # Sort matches in descending order of score
            m_sort = cos_scores.argsort()[::-1]
            matches = matches[m_sort]

            if return_united:
                # Save cos scores for later return
                cos_scores_df = pd.DataFrame(matches,columns=['i','j'])
                cos_scores_df['cos'] = cos_scores[m_sort]

            # Set up tensors
            matches = torch.tensor(matches).to(self.device)
            
            # Set-up per-string tracking of never-match relationships
            if never_match is not None:
                never_match_map = {s:sep for sep in never_match for s in sep}
                
                if always_match is not None:
                    # If always_match, we use group labels instead of the strings themselves
                    never_match_array = np.array([never_match_map.get(always_match_labels[s],set()) for s in self.strings])
                else:
                    never_match_array = np.array([never_match_map.get(s,set()) for s in self.strings])                   
            

            n_matches = matches.shape[0]
            with tqdm(total=n_matches,desc='Uniting matches',
                        delay=1,disable=not progress_bar) as p_bar:

                while len(matches):

                    # Select the current match pair and remove it from the queue
                    match_pair = matches[0]
                    matches = matches[1:]

                    # Get the groups of the current match pair
                    g = group_ids[match_pair]
                    g0 = group_ids == g[0]
                    g1 = group_ids == g[1]

                    # Identify which strings should be united
                    to_unite = g0 | g1

                    # Flag whether the new group will have three or more strings
                    singletons = to_unite.sum() < 3

                    # Start by asuming that we can match this pair
                    unite_ok = True

                    # Check whether uniting this pair will unite any never_match strings/labels
                    if never_match is not None:
                        never_0 = never_match_array[match_pair[0]]
                        never_1 = never_match_array[match_pair[1]]

                        if never_0 and never_1 and (never_0 & never_1):
                            # Here we make use of the fact that any pair of never_match strings/labels
                            # will appear in both never_0 and never_1 if one string/label is in each group
                            unite_ok = False

                    # Check whether the uniting the pair will violate the group_threshold
                    # (impossible if the strings are singletons)
                    if unite_ok and group_threshold and not singletons:
                        V0 = V[g0,:]
                        V1 = V[g1,:]

                        unite_ok = (V0@V1.T).min() >= separate_cos


                    if unite_ok:

                        # Unite groups
                        group_ids[to_unite] = g[0]

                        if never_match and (never_0 or never_1):
                            # Propagate never_match information to the whole group
                            never_match_array[to_unite.detach().cpu().numpy()] = never_0 | never_1
                            
                        # If we are uniting more than two strings, we can eliminate
                        # some redundant matches in the queue
                        if not singletons:
                            # Removed queued matches that are now in the same group
                            matches = matches[group_ids[matches[:,0]] != group_ids[matches[:,1]]]

                        if return_united:
                            match_record = np.empty(4,dtype=int)
                            match_record[:2] = match_pair.cpu().numpy().ravel()
                            match_record[2] = self.counts[g0].sum().item()
                            match_record[3] = self.counts[g1].sum().item()
                            
                            united.append(match_record)
                    else:
                        # Remove queued matches connecting these groups
                        matches = matches[torch.isin(group_ids[matches[:,0]],g,invert=True) \
                                            | torch.isin(group_ids[matches[:,1]],g,invert=True)]

                    # Update progress bar
                    p_bar.update(n_matches - matches.shape[0])
                    n_matches = matches.shape[0]

        predicted_grouping = self.ids_to_group(group_ids)

        if always_match is not None:
            predicted_grouping = predicted_grouping.unite(always_grouping)

        if return_united:
            united_df = pd.DataFrame(np.vstack(united),columns=['i','j','n_i','n_j'])
            united_df = pd.merge(united_df,cos_scores_df,how='inner',on=['i','j'])
            united_df['score'] = self.score_model(
                                    torch.tensor(united_df['cos'].values).to(self.device)
                                    ).cpu().numpy()
            
            united_df = united_df.drop('cos',axis=1)
            
            for c in ['i','j']:
                united_df[c] = [self.strings[i] for i in united_df[c]]

            if always_match is not None:
                united_df['always_match'] = [always_grouping[i] == always_grouping[j] 
                                            for i,j in united_df[['i','j']].values]

            return predicted_grouping,united_df
            
        else:

            return predicted_grouping

    @torch.no_grad()
    def unite_nearest(self,target_strings,threshold=0,always_grouping=None,progress_bar=True,batch_size=64):
        """
        Unite embedding strings with each string's most similar target string.

        - "always_grouping" will be used to inialize the group_ids before uniting new matches
        - "theshold" sets the minimimum match similarity required between a string and target string
          for the string to be matched. (i.e., setting theshold=0 will result in every embedding
          string to be matched its nearest target string, while setting threshold=0.9 will leave
          strings that have similarity<0.9 with their nearest target string unaffected)

        returns: MatchGroups object
        """

        if always_grouping is not None:
            # self = self.embed(always_grouping)
            group_ids = self._group_to_ids(always_grouping)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)

        seed_ids = torch.tensor([self.string_map[s] for s in target_strings]).to(self.device)
        V_seed = V[seed_ids]
        g_seed = group_ids[seed_ids]
        is_seed = torch.zeros(V.shape[0],dtype=torch.bool).to(self.device)
        is_seed[g_seed] = True

        for batch_start in tqdm(range(0,len(self),batch_size),
                                    delay=1,desc='Predicting matches',disable=not progress_bar):

            batch_slice = slice(batch_start,batch_start+batch_size)

            batch_cos = V[batch_slice]@V_seed.T

            max_cos,max_seed = torch.max(batch_cos,dim=1)

            # Get batch index locations where score > threshold
            batch_i = torch.nonzero(max_cos > cos_threshold)

            if len(batch_i):
                # Drop target strings from matches (otherwise numerical precision
                # issues can allow target strings to match to other strings)
                batch_i = batch_i[~is_seed[batch_slice][batch_i]]

                if len(batch_i):
                    # Get indices of matched strings
                    i = batch_i + batch_start

                    # Assign matched strings to the target string's group
                    group_ids[i] = g_seed[max_seed[batch_i]]

        return self._ids_to_group(group_ids)

    @torch.no_grad()
    def score_pairs(self,string_pairs,batch_size=64,progress_bar=True):
        string_pairs = np.array(string_pairs)

        scores = []
        for batch_start in tqdm(range(0,string_pairs.shape[0],batch_size),desc='Scoring pairs',disable=not progress_bar):

            V0 = self[string_pairs[batch_start:batch_start+batch_size,0]].V
            V1 = self[string_pairs[batch_start:batch_start+batch_size,1]].V

            batch_cos = (V0*V1).sum(dim=1).ravel()
            batch_scores = self.score_model(batch_cos)
            
            scores.append(batch_scores.cpu().numpy())

        return np.concatenate(scores)

    @torch.no_grad()
    def _batch_scores(self,group_ids,batch_start,batch_size,
                            is_match=None,
                            min_score=None,max_score=None,
                            min_loss=None,max_loss=None):

        strings = self.strings
        V = self.V
        w = self.w

        # Create simple slice objects to avoid creating copies with advanced indexing
        i_slice = slice(batch_start,batch_start+batch_size)
        j_slice = slice(batch_start+1,None)

        X = V[i_slice]@V[j_slice].T
        Y = (group_ids[i_slice,None] == group_ids[None,j_slice]).float()
        if w is not None:
            W = w[i_slice,None]*w[None,j_slice]
        else:
            W = None

        scores = self.score_model(X)
        loss = self.score_model.loss(X,Y,weights=W)

        # Search upper diagonal entries only
        # (note j_slice starting index is offset by one)
        scores = torch.triu(scores)

        # Filter by match type
        if is_match is not None:
            if is_match:
                scores *= Y
            else:
                scores *= (1 - Y)

        # Filter by min score
        if min_score is not None:
            scores *= (scores >= min_score)

        # Filter by max score
        if max_score is not None:
            scores *= (scores <= max_score)

        # Filter by min loss
        if min_loss is not None:
            scores *= (loss >= min_loss)

        # Filter by max loss
        if max_loss is not None:
            scores *= (loss <= max_loss)

        # Collect scored pairs
        i,j = torch.nonzero(scores,as_tuple=True)

        pairs = np.hstack([
                                strings[i.cpu().numpy() + batch_start][:,None],
                                strings[j.cpu().numpy() + (batch_start + 1)][:,None]
                                ])

        pair_groups = np.hstack([
                                strings[group_ids[i + batch_start].cpu().numpy()][:,None],
                                strings[group_ids[j + (batch_start + 1)].cpu().numpy()][:,None]
                                ])

        pair_scores = scores[i,j].cpu().numpy()
        pair_losses = loss[i,j].cpu().numpy()

        return pairs,pair_groups,pair_scores,pair_losses

    def iter_scores(self,grouping=None,batch_size=64,progress_bar=True,**kwargs):

        if grouping is not None:
            self = self.embed(grouping)
            group_ids = self._group_to_ids(grouping)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        for batch_start in tqdm(range(0,len(self),batch_size),desc='Scoring pairs',disable=not progress_bar):
            pairs,pair_groups,scores,losses = self._batch_scored_pairs(self,group_ids,batch_start,batch_size,**kwargs)
            for (s0,s1),(g0,g1),score,loss in zip(pairs,pair_groups,scores,losses):
                yield {
                        'string0':s0,
                        'string1':s1,
                        'group0':g0,
                        'group1':g1,
                        'score':score,
                        'loss':loss,
                        }


def load_embeddings(f):
    """
    Load embeddings from custom zipped archive format
    """
    with ZipFile(f,'r') as zip:
        score_model = pickle.loads(zip.read('score_model.pkl'))
        weighting_function = pickle.loads(zip.read('weighting_function.pkl'))
        strings_df = pd.read_csv(zip.open('strings.csv'),na_filter=False)
        V = np.load(zip.open('V.npy'))

        return Embeddings(
                            strings=strings_df['string'].values,
                            counts=torch.tensor(strings_df['count'].values),
                            score_model=score_model,
                            weighting_function=weighting_function,
                            V=torch.tensor(V)
                            )

