from cliMLe.learning import LearningModel
from cliMLe.analytics import Visualizer

instances = [ "ams-zg-16-16-" + str( index ) for index in range(1,11)]
target_values = [3.0, -3.0]

results = {}
for iT, target_value in enumerate(target_values):
    for iI, instance in enumerate(instances):
        learningModel, model, weights = LearningModel.getActivationBackProjection(instance, target_value)
        patterns = Visualizer.getWeightedPatterns(learningModel, weights)
        if iI == 0:
            results[iT] = patterns[0]
        else:
            results[iT] = results[iT] + patterns[0]

results[2] = results[0] - results[1]
Visualizer.mpl_spaceplot(results.values(), ["500 mbar Height - Strong Monsoon", "500 mbar Height - Weak Monsoon", "Difference"] )
