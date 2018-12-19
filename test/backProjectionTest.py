from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel, ActivationTarget
import matplotlib.pyplot as plt



strong_monsoon = ActivationTarget( 3.0, "Strong Monsoon" )
weak_monsoon = ActivationTarget( -3.0, "Weak Monsoon" )

instance = "ams-2"
activation_target = strong_monsoon
learningModel, model, result = LearningModel.getActivationBackProjection( instance, activation_target.value )
inputs = learningModel.inputs

fig, ax = plt.subplots()
y_pos = np.arange(result.shape[0])
plt.bar( y_pos, result , align='center', alpha=0.5)
plt.ylabel('Activation')
plt.title( activation_target.title + ' Activtion Plot')
plt.show()