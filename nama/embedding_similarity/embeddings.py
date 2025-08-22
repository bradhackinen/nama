import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import copy
from collections import Counter
import torch
from zipfile import ZipFile
import pickle
from io import BytesIO

from ..match_data import MatchData
from ..utils import check_always_never_match_conflicts


class Embeddings(torch.nn.Module):
    """
    Stores embeddings for a fixed array of strings and provides methods for
    clustering the strings to create MatchData objects according to different
    algorithms.
    """
    def __init__(self,strings,counts,V,score_model,device='cpu'):
        super().__init__()

        self.strings = np.array(list(strings))
        self.string_map = {s:i for i,s in enumerate(strings)}
        self.V = V
        self.counts = counts
        # self.w = weighting_function(counts)
        self.score_model = score_model
        # self.weighting_function = weighting_function
        self.device = device

        self.to(device)

    def __repr__(self):
        return f'<nama.Embeddings containing {self.V.shape[1]}-d vectors for {len(self)} strings'

    def to(self,device):
        super().to(device)
        self.V = self.V.to(device)
        self.counts = self.counts.to(device)
        # self.w = self.w.to(device)
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

            # # Write score model
            # zip.writestr('weighting_function.pkl',pickle.dumps(self.weighting_function))

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
        elif isinstance(arg, MatchData):
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
        # new.w = self.w[i]
        new.string_map = {s:i for i,s in enumerate(new.strings)}

        return new

    def embed(self,matches):
        """
        Construct updated Embeddings with counts from the input MatchData
        """
        new = self[matches]
        new.counts = torch.tensor([matches.counts[s] for s in new.strings],device=self.device)
        new.w = new.weighting_function(new.counts)

        return new

    def __len__(self):
        return len(self.strings)

    def _match_data_to_group_ids(self,matches):
        group_id_map = {g:i for i,g in enumerate(matches.groups.keys())}
        group_ids = torch.tensor([group_id_map[matches[s]] for s in self.strings]).to(self.device)
        return group_ids

    def _group_ids_to_match_data(self,group_ids):
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

        # Build the matches
        matches = MatchData()
        matches.counts = Counter({s:int(c) for s,c in zip(strings,counts)})
        matches.labels = {s:g[-1] for g in groups for s in g}
        matches.groups = {g[-1]:list(g) for g in groups}

        return matches

    @torch.no_grad()
    def _fast_unite_similar(self,group_ids,threshold=0.5,progress_bar=True,batch_size=64):

        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)

        for batch_start in tqdm(range(0,len(self),batch_size),
                                    desc='Predicting matches',delay=1,smoothing=0.01,
                                    disable=not progress_bar):

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

        return self._group_ids_to_match_data(group_ids)
    
    def _get_ordered_match_pairs(self,group_ids=None,threshold=0.5,batch_size=64,progress_bar=True):
        """
        Get all pairs of strings with similarity > threshold ordered by similarity (descending).
        """
        if group_ids is None:
            group_ids = torch.arange(len(self)).to(self.device)

        # Convert thresholds from scores to raw cosine distances
        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)

        # First collect all pairs to match (can be memory intensive!)
        matches = []
        cos_scores = []
        for batch_start in tqdm(range(0,len(self),batch_size),
                                    desc='Scoring pairs',delay=1,smoothing=0.01,
                                    disable=not progress_bar):

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

                # Can skip strings that are already matched in the base matches
                unmatched = group_ids[i] != group_ids[j]
                i = i[unmatched]
                j = j[unmatched]
                cos = cos[unmatched]

                if len(i):
                    batch_matches = torch.hstack([i[:,None],j[:,None]])

                    matches.append(batch_matches.to('cpu').numpy())
                    cos_scores.append(cos.to('cpu').numpy())

        if not matches:
            return torch.empty(0,2),torch.empty(0)

        matches = np.vstack(matches)
        cos_scores = np.hstack(cos_scores).T

        # Sort matches in descending order of score
        m_sort = cos_scores.argsort()[::-1]
        matches = matches[m_sort]
        cos_scores = cos_scores[m_sort]

        return matches,cos_scores
    
    def _unite_ordered_match_pairs(self,matches,group_ids=None,never_match=None,always_match=None,
                                   group_threshold=None,min_loss=False,
                                   return_united=False,progress_bar=True):

        # Convert thresholds from scores to raw cosine distances
        if group_threshold is not None:
            group_threshold_cos = self.score_model.score_to_cos(group_threshold)

        if min_loss:
            # convert loss_threshold to reference loss level
            # loss_threshold_cos = self.score_model.score_to_cos(loss_threshold)
            # loss_threshold = self.score_model.loss(torch.tensor(loss_threshold_cos),0.5)
            loss_function = self.score_model.loss

            

        # Set up tensors
        V = self.V
        matches = torch.tensor(matches).to(self.device)
        
        # Set-up per-string tracking of never-match relationships
        # Note: Each string can be part of multiple never-match sets
        if never_match is not None:
            never_match_strings = set.union(*never_match)
            never_match_map = {s:set() for s in never_match_strings}

            for g in never_match:
                for s in g:
                    never_match_map[s] |= g
            
            if always_match is not None:
                # If always_match, we use group labels instead of the strings themselves
                always_match_labels = always_match.labels
                never_match_array = np.array([never_match_map.get(always_match_labels[s],set()) for s in self.strings])
            else:
                never_match_array = np.array([never_match_map.get(s,set()) for s in self.strings])                   
        

        united = []
        n_matches = matches.shape[0]
        with tqdm(total=n_matches,
                    desc='Uniting matches',delay=1,smoothing=0.01,
                    disable=not progress_bar) as p_bar:

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
                X = None
                if unite_ok and group_threshold and not singletons:

                    V0 = V[g0,:]
                    V1 = V[g1,:]
                    
                    X = V0@V1.T
                    
                    unite_ok = X.min() >= group_threshold_cos

                if unite_ok and min_loss:

                    # pass
                    
                    # Avoid re-computing X if we already have it
                    if X is None:
                        V0 = V[g0,:]
                        V1 = V[g1,:]

                        X = V0@V1.T

                    unite_loss = loss_function(X,torch.ones_like(X,device=self.device)).mean()
                    sep_loss = loss_function(X,torch.zeros_like(X,device=self.device)).mean()

                    unite_ok = unite_loss < sep_loss

                    # unite_ok = unite_loss - sep_loss < loss_threshold

                    # v0 = V0.mean(dim=0)
                    # v1 = V1.mean(dim=0)


                    # v_unite = V0.shape[0] * v0 + V1.shape[0] * v1

                    # v0 = v0 / torch.sqrt((v0**2).sum())
                    # v1 = v1 / torch.sqrt((v1**2).sum())
                    # v_unite = v_unite / torch.sqrt((v_unite**2).sum())
                    
                    # sep_loss = (loss_function(V0@v0.T,1).sum() + loss_function(V0@v1.T,0).sum() 
                    #             + loss_function(v1@v1.T,1).sum() + loss_function(V1@v0.T,0).sum())
                    
                    # unite_loss = 2*(loss_function(V0@v_unite.T,1).sum() + loss_function(V1@v_unite.T,1).sum())

                    
                    # unite_ok = unite_loss < sep_loss

                    # # loss_delta = 

                    # unite_ok = loss_function(X,1).sum() < loss_function(X,0).sum() + loss_threshold
                    # # unite_ok = loss_function(X,1).mean() < loss_threshold


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

        predicted_matches = self._group_ids_to_match_data(group_ids)

        if always_match is not None:
            predicted_matches = predicted_matches.unite(always_match)

        if return_united:
            return predicted_matches,united
        else:
            return predicted_matches


    @torch.no_grad()
    def unite_similar(self,
                threshold=0.5,
                group_threshold=None,
                min_loss=None,
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
        - "min_loss" changes the objective function to only unite strings if the loss to 
            separate them is greater than the loss to unite them. This approximates 
            minimizing the total pairwise loss for the predicted groups.
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
        result in matches any two strings with similarity<group_threshold. If so, this
        pair is skipped. This version of the algorithm requires more memory and processing
        time, but guaruntees deterministic output that is consistent with the constraints.
            
        returns: MatchData object
        """
        # if group_threshold and group_threshold < threshold:
        #     raise ValueError('group_threshold must be greater than or equal to threshold')

        group_ids = torch.arange(len(self)).to(self.device)
        
        if always_match is not None:
            # Ensure that always_match is a mathc data object
            # With no strings not in the embeddings
            always_match = (MatchData(self.strings)
                            .unite(always_match))
            always_match_labels = always_match.labels


        # Use a simpler, faster prediction algorithm if possible
        if not (return_united or group_threshold or min_loss or (never_match is not None)):
            if always_match is not None:
                group_ids = self._match_data_to_group_ids(always_match)

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
                check_always_never_match_conflicts(always_match,never_match,
                                                    always_never_conflicts=always_never_conflicts)

                # If always_match, collapse to group labels that should not match
                # Note: Implicitly letting always_match override never_match here
                never_match = [{always_match_labels[s] for s in g if s in always_match_labels} for g in never_match]
            
            else:
                # Otherwise just use the strings themselves as labels
                never_match = [set(s) for s in never_match]

        # Convert thresholds from scores to raw cosine distances
        

        # First collect all pairs to match (can be memory intensive!)
        matches, cos_scores = self._get_ordered_match_pairs(
                                            group_ids=group_ids,
                                            threshold=threshold,
                                            batch_size=batch_size,
                                            progress_bar=progress_bar)

        # Unite potential match pairs in priority order, while respecting
        # the group_threshold and never_match arguments
        if matches.shape[0] > 0:
            if return_united:
                # Save cos scores for later return
                cos_scores_df = pd.DataFrame(matches,columns=['i','j'])
                cos_scores_df['cos'] = cos_scores

                predicted_matches,united = self._unite_ordered_match_pairs(
                                            matches=matches,
                                            group_ids=group_ids,
                                            never_match=never_match,
                                            group_threshold=group_threshold,
                                            min_loss=min_loss,
                                            return_united=True,
                                            progress_bar=progress_bar)
                
            else:
                predicted_matches = self._unite_ordered_match_pairs(
                                            matches=matches,
                                            group_ids=group_ids,
                                            never_match=never_match,
                                            group_threshold=group_threshold,
                                            min_loss=min_loss,
                                            progress_bar=progress_bar)

        else:
            predicted_matches = self._group_ids_to_match_data(group_ids)
            united = []
        
        if return_united:
            if len(united):
                united_df = pd.DataFrame(np.vstack(united),columns=['i','j','n_i','n_j'])
                united_df = pd.merge(united_df,cos_scores_df,how='inner',on=['i','j'])
                united_df['score'] = self.score_model(
                                        torch.tensor(united_df['cos'].values).to(self.device)
                                        ).cpu().numpy()
                
                united_df = united_df.drop('cos',axis=1)
                
                for c in ['i','j']:
                    united_df[c] = [self.strings[i] for i in united_df[c]]

                if always_match is not None:
                    united_df['always_match'] = [always_match[i] == always_match[j] 
                                                for i,j in united_df[['i','j']].values]
            else:
                if always_match is not None:
                    united_df = pd.DataFrame(columns=['i','j','n_i','n_j','score','always_match'])
                else:
                    united_df = pd.DataFrame(columns=['i','j','n_i','n_j','score'])

            return predicted_matches,united_df
        
        else:
            return predicted_matches



    @torch.no_grad()
    def unite_nearest(self,target_strings,threshold=0,always_match=None,progress_bar=True,batch_size=64):
        """
        Unite embedding strings with each string's most similar target string.

        - "always_matches" will be used to inialize the group_ids before uniting new matches
        - "theshold" sets the minimimum match similarity required between a string and target string
          for the string to be matched. (i.e., setting theshold=0 will result in every embedding
          string to be matched its nearest target string, while setting threshold=0.9 will leave
          strings that have similarity<0.9 with their nearest target string unaffected)

        returns: MatchData object
        """

        if always_match is not None:
            # self = self.embed(always_matches)
            group_ids = self._match_data_to_group_ids(always_match)
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
                                    desc='Predicting matches',delay=1,smoothing=0.01,
                                    disable=not progress_bar):

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

        return self._group_ids_to_match_data(group_ids)



    # def inner_join(self,left_strings,right_strings,threshold=0,progress_bar=True,batch_size=64):
    #     """
    #     Match left strings to right strings, and return a dataframe of scored pairs.

    #     - "theshold" sets the minimimum match similarity required between the left and right strings.

    #     returns: pd.DataFrame with columns ['i','j','score']
    #     """

    #     V = self.V
    #     cos_threshold = self.score_model.score_to_cos(threshold)

    #     left_ids = torch.tensor([self.string_map[s] for s in left_strings]).to(self.device)
    #     right_ids = torch.tensor([self.string_map[s] for s in right_strings]).to(self.device)

    #     left_V = V[left_ids]
    #     right_V = V[right_ids]

    #     pairs = []
    #     scores = []
    #     for batch_start in tqdm(range(0,len(left_strings),batch_size),
    #                                 desc='Joining Pairs',delay=1,smoothing=0.01,
    #                                 disable=not progress_bar):

    #         batch_slice = slice(batch_start,batch_start+batch_size)

    #         batch_cos = left_V[batch_slice]@right_V.T

    #         batch_i,batch_j = torch.nonzero(batch_cos > cos_threshold,as_tuple=True)

    #         if len(batch_i):
    #             # Get indices of matched strings
    #             i = batch_i + batch_start
    #             j = batch_j

    #             # Assign matched strings to the target string's group
    #             left_group_ids[i] = right_group_ids[j]

    #             batch_scores = self.score_model(batch_cos[batch_i,batch_j])

    #             scores.append(batch_scores.cpu().numpy())

    #     scores = np.hstack(scores)

    #     return pd.DataFrame({
    #                         'i':left_strings,
    #                         'j':right_strings,
    #                         'score':scores,
    #                         })


    

    @torch.no_grad()
    def score_pairs(self,string_pairs,batch_size=64,progress_bar=True):
        string_pairs = np.array(string_pairs)

        scores = []
        for batch_start in tqdm(range(0,string_pairs.shape[0],batch_size),
                                desc='Scoring pairs',delay=1,smoothing=0.01,
                                disable=not progress_bar):

            V0 = self[string_pairs[batch_start:batch_start+batch_size,0]].V
            V1 = self[string_pairs[batch_start:batch_start+batch_size,1]].V

            batch_cos = (V0*V1).sum(dim=1).ravel()
            batch_scores = self.score_model(batch_cos)
            
            scores.append(batch_scores.cpu().numpy())

        return np.concatenate(scores)

    # @torch.no_grad()
    # def _batch_scores(self,group_ids,batch_start,batch_size,
    #                         is_match=None,
    #                         min_score=None,max_score=None,
    #                         min_loss=None,max_loss=None):

    #     strings = self.strings
    #     V = self.V
    #     w = self.w

    #     # Create simple slice objects to avoid creating copies with advanced indexing
    #     i_slice = slice(batch_start,batch_start+batch_size)
    #     j_slice = slice(batch_start+1,None)

    #     X = V[i_slice]@V[j_slice].T
    #     Y = (group_ids[i_slice,None] == group_ids[None,j_slice]).float()
    #     if w is not None:
    #         W = w[i_slice,None]*w[None,j_slice]
    #     else:
    #         W = None

    #     scores = self.score_model(X)
    #     loss = self.score_model.loss(X,Y,weights=W)

    #     # Search upper diagonal entries only
    #     # (note j_slice starting index is offset by one)
    #     scores = torch.triu(scores)

    #     # Filter by match type
    #     if is_match is not None:
    #         if is_match:
    #             scores *= Y
    #         else:
    #             scores *= (1 - Y)

    #     # Filter by min score
    #     if min_score is not None:
    #         scores *= (scores >= min_score)

    #     # Filter by max score
    #     if max_score is not None:
    #         scores *= (scores <= max_score)

    #     # Filter by min loss
    #     if min_loss is not None:
    #         scores *= (loss >= min_loss)

    #     # Filter by max loss
    #     if max_loss is not None:
    #         scores *= (loss <= max_loss)

    #     # Collect scored pairs
    #     i,j = torch.nonzero(scores,as_tuple=True)

    #     pairs = np.hstack([
    #                             strings[i.cpu().numpy() + batch_start][:,None],
    #                             strings[j.cpu().numpy() + (batch_start + 1)][:,None]
    #                             ])

    #     pair_groups = np.hstack([
    #                             strings[group_ids[i + batch_start].cpu().numpy()][:,None],
    #                             strings[group_ids[j + (batch_start + 1)].cpu().numpy()][:,None]
    #                             ])

    #     pair_scores = scores[i,j].cpu().numpy()
    #     pair_losses = loss[i,j].cpu().numpy()

    #     return pairs,pair_groups,pair_scores,pair_losses
    

    @torch.no_grad()
    def get_mean_loss(self,matches,batch_size=64,progress_bar=True,weight_by_counts=False,weight_exponent=0.5,**kwargs):
        """
        Get the mean loss for all pairs of strings in the embeddings
        """
        if matches is not None:
            self = self.embed(matches)
            group_ids = self._match_data_to_group_ids(matches)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        V = self.V

        if weight_by_counts:
            w = self.counts**weight_exponent
        else:
            w = None
        
        # Get the mean loss for all pairs of strings in the embeddings
        batch_losses = []
        batch_sizes = []
        for batch_start in tqdm(range(0,len(self),batch_size),
                                    desc='Scoring pairs',delay=1,smoothing=0.01,
                                    disable=not progress_bar):

            V_i = V[batch_start:batch_start+batch_size]

            X = V_i@V.T
            Y = (group_ids[batch_start:batch_start+batch_size,None] == group_ids[None,:]).float()
            
            if w is not None:
                W = w[batch_start:batch_start+batch_size,None]*w[None,:]
            else:
                W = None

            pairwise_loss = self.score_model.loss(X,Y,weights=W)

            batch_losses.append(pairwise_loss.sum().item())
            batch_sizes.append(pairwise_loss.shape[0]*pairwise_loss.shape[1])

        total_loss = sum(batch_losses)
        total_size = sum(batch_sizes)

        mean_loss = total_loss / total_size

        return mean_loss


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

