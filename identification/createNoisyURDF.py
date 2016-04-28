
import numpy as np
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
import identificationHelpers
import argparse

def main():
    parser = argparse.ArgumentParser(description='Load measurements and URDF model to get inertial parameters.')
    parser.add_argument('--urdf_input', required=True, type=str, help='the file to load the robot model from')
    parser.add_argument('--urdf_output', required=True, type=str, help='the file to save the noisy robot model to')
    args = parser.parse_args()

    model = iDynTree.Model()
    iDynTree.modelFromURDF(args.urdf_input, model)
    link_names = []
    for i in range(0, model.getNrOfLinks()):
        link_names.append(model.getLinkName(i))
    n_params = model.getNrOfLinks()*10

    dynComp = iDynTree.DynamicsComputations()
    dynComp.loadRobotModelFromFile(args.urdf_input)
    xStdModel = iDynTree.VectorDynSize(n_params)
    dynComp.getModelDynamicsParameters(xStdModel)
    xStdModel = xStdModel.toNumPy()
    xStdModel += np.random.randn(n_params)*0.01

    helpers = identificationHelpers.IdentificationHelpers(n_params)
    helpers.replaceParamsInURDF(input_urdf=args.urdf_input, output_urdf=args.urdf_output, \
                                params=xStdModel, link_names=link_names)

if __name__ == '__main__':
    main()
