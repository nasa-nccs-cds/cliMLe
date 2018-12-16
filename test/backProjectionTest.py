from cliMLe.trainingData import *
from cliMLe.learning import FitResult, LearningModel
import matplotlib.pyplot as plt

instance = "ams-result-1"
result = LearningModel.getBackProjection( instance )

fig, ax = plt.subplots()
y_pos = np.arange(result.shape[0])
plt.bar( y_pos, result , align='center', alpha=0.5)
plt.ylabel('Activation')
plt.title('Max Activtion Plot')
plt.show()