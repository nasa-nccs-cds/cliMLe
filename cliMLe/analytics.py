from cliMLe.learning import LearningModel, ActivationTarget
from cliMLe.climatele import Experiment, ClimateleDataset
from cdms2.tvariable import TransientVariable
from mpl_toolkits.basemap import Basemap
from cliMLe.project import PC, EOF
from cliMLe.svd import Eof
import matplotlib.pyplot as plt
import numpy as np
import cdms2

class Visualizer:

    @classmethod
    def getModes( cls, lmod, nModes ):
        # type: (LearningModel,int) -> np.array
        solver = Eof( lmod.inputData )
        modes = solver.eofs(neofs=nModes)
        return modes

    @classmethod
    def getPCDistribution( cls, lmod ):
        # type: (LearningModel) -> ( np.array, np.array, np.array )
        modes = cls.getModes(lmod,2)
        x, y, c = [], [], []
        for iPoint in range( modes.shape[1] ):
            input = lmod.inputData[iPoint]
            x.append( np.dot( modes[0], input ) )
            y.append( np.dot(modes[1], input) )
            c.append( lmod.outputData[iPoint][0] )
        return np.array( x ), np.array( y ), np.array( c )

    @classmethod
    def plotModeDistribution( cls, lmod ):
        # type: (LearningModel) -> None
        x, y, c = Visualizer.getPCDistribution(lmod)
        fig, ax = plt.subplots()
        cax = plt.scatter( x, y, None, c, None, "jet" )
        cbar = fig.colorbar( cax )
        plt.title( ' PC Distribution Plot ' )
        plt.show()

    @classmethod
    def mpl_spaceplot( cls, variables, smooth=True, cbar = False ):
        # type: ( str, List[TransientVariable], bool ) -> None
        fig = plt.figure()
        nvars = len(variables)
        for ivar in range(nvars):
            variable = variables[ivar]
            ax = fig.add_subplot( nvars, 1, ivar+1 )
            ax.set_title(variable.id)
            lons = variable.getLongitude().getValue()
            lats = variable.getLatitude().getValue()
            m = Basemap( llcrnrlon=lons[0],
                         llcrnrlat=lats[0],
                         urcrnrlon=lons[len(lons)-1],
                         urcrnrlat=lats[len(lats)-1],
                         epsg='4326',
                         lat_0 = lats.mean(),
                         lon_0 = lons.mean())
            lon, lat = np.meshgrid( lons, lats )
            xi, yi = m(lon, lat)
            smoothing = 'gouraud' if smooth else 'flat'
            print " Plotting variable: " + variable.id + ", shape = " + str(variable.shape)
            cs2 = m.pcolormesh(xi, yi, variable.getValue(), cmap='jet', shading=smoothing )
            lats_space = abs(lats[0])+abs(lats[len(lats)-1])
            m.drawparallels(np.arange(lats[0],lats[len(lats)-1], round(lats_space/5, 0)), labels=[1,0,0,0], dashes=[6,900])
            lons_space = abs(lons[0])+abs(lons[len(lons)-1])
            m.drawmeridians(np.arange(lons[0],lons[len(lons)-1], round(lons_space/5, 0)), labels=[0,0,0,1], dashes=[6,900])
            m.drawcoastlines()
            m.drawstates()
            m.drawcountries()
            if cbar: m.colorbar(cs2,pad="10%")
        plt.show()

    @classmethod
    def getWeightedPatterns( cls, lmod, weights ):
        # type: ( LearningModel, np.array ) -> List[ cdms2.tvariable.TransientVariable ]
        results = {}
        for cdset in lmod.inputs.sources:
            variables = cdset.getVariables(EOF)
            for ivar in range(len(variables)):
                variable = variables[ivar]  # type: cdms2.tvariable.TransientVariable
                attrs = variable.attributes['long_name'].split(",")
                exp = cdset.getExperiment(attrs[1])
                dset = exp.getDataset(EOF)
                eof = dset[attrs[0]]
                weight =  weights[ ivar ] * 25.0
                result = results.get( attrs[1], None )
                new_result = weight * eof if result is None else result + weight * eof
                results[attrs[1]] = TransientVariable( new_result, axes=[eof.getLatitude(),eof.getLongitude()], id=exp.id )
        return results.values()


if __name__ == "__main__":
    instance = "good_2"
    target_value = 3.0
    learningModel, model, weights = LearningModel.getActivationBackProjection( instance,  target_value )
    patterns = Visualizer.getWeightedPatterns( learningModel, weights )
    Visualizer.mpl_spaceplot( patterns )
