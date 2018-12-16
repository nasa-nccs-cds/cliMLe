from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
import matplotlib.pyplot as plt

class ActivationTarget:
    def __init__(self, target_value, title_str ):
        self.value = target_value
        self.title = title_str

strong_monsoon = ActivationTarget( 3.0, "Strong Monsoon" )
weak_monsoon = ActivationTarget( -3.0, "Weak Monsoon" )

instance = "ams-result-1"
activation_target = strong_monsoon
result = LearningModel.getActivationBackProjection( instance, activation_target.value )

fig, ax = plt.subplots()
y_pos = np.arange(result.shape[0])
plt.bar( y_pos, result , align='center', alpha=0.5)
plt.ylabel('Activation')
plt.title( activation_target.title + ' Activtion Plot')
plt.show()