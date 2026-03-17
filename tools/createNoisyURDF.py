#!/usr/bin/env python

import numpy as np
from idyntree import bindings as iDynTree
import identificationHelpers
import argparse


def main():
    """
    open a urdf file and add noise to each parameter. Can be used for testing, but noisy params are usually not consistent, so may be of limited use.
    """
    parser = argparse.ArgumentParser(
        description="Load measurements and URDF model to get inertial parameters."
    )
    parser.add_argument(
        "--urdf_input",
        required=True,
        type=str,
        help="the file to load the robot model from",
    )
    parser.add_argument(
        "--urdf_output",
        required=True,
        type=str,
        help="the file to save the noisy robot model to",
    )
    parser.add_argument(
        "--noise", required=False, type=float, help="scale of noise (default 0.01)"
    )
    parser.set_defaults(noise=0.01)
    args = parser.parse_args()

    loader = iDynTree.ModelLoader()
    loader.loadModelFromFile(args.urdf_input)
    model = loader.model()
    link_names = []
    for i in range(0, model.getNrOfLinks()):
        link_names.append(model.getLinkName(i))
    n_params = model.getNrOfLinks() * 10

    xStdModel = iDynTree.VectorDynSize(n_params)
    model.getInertialParameters(xStdModel)
    xStdModel = xStdModel.toNumPy()
    # percentage noise
    # for p in range(0, len(xStdModel)):
    #    xStdModel[p] += np.random.randn()*args.noise*xStdModel[p]
    # additive noise
    xStdModel += np.random.randn(n_params) * args.noise

    helpers = identificationHelpers.IdentificationHelpers(n_params)
    helpers.replaceParamsInURDF(
        input_urdf=args.urdf_input,
        output_urdf=args.urdf_output,
        new_params=xStdModel,
        link_names=link_names,
    )


if __name__ == "__main__":
    main()
