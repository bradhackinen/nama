import nama
from nama.matcher import Matcher
from nama.rnnEmbedding import RnnEmbeddingModel

from nama.defaults import *


# Initialize the matcher
matcher = Matcher(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO','ABC Inc.','XYZ Co.'])

# Add some corpHash matches
matcher.matchHash(nama.hashes.corpHash)


# Initalize a new, untrained similarity model (gpu accelerated with device='cuda')
similarityModel = RnnEmbeddingModel(d=100,d_recurrent=100,recurrent_layers=2,bidirectional=True)


# Observe that untrained suggestions are poor quality (though not entirely useless - neat!)
matcher.suggestMatches(similarityModel)


# Train model using existing matches
similarityModel.train(matcher,epochs=1)

# Suggestions are now much better
matcher.suggestMatches(similarityModel)

# Save similarity model
similarityModel.save(os.path.join(modelDir,'demoModel.bin'))


# Too much training on a small set of strings can lead to over-fitting
similarityModel.train(matcher,epochs=10)

# --> All suggestions now have very low scores
matcher.suggestMatches(similarityModel)


# Note: On a larger set of strings, it is important to train for a much longer time
# But the best training amount is unclear (hours or days for very large datasets)
# Total training = n_epochs*epoch_size
# Results will probably depend on details of the data and the settings of the rnnEmbeddingModel
