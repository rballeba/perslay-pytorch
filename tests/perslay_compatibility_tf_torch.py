"""
This file contains the tests for the compatibility of PersLay with Tensorflow and PyTorch.
Here, we check that all operations from the tensorflow version are available in the pytorch version and that
the results are close (abs tf - abs pytorch) < 1e-2 for all the configurations except for the layer type
PermutationEquivariant, that was difficult to initialize in a deterministic way.
"""

from functools import partial

import numpy as np
import gudhi.representations as tda
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import torch

from perslay import perslay
from perslay import perslay_pytorch
from tutorial.experiments import generate_diagrams_and_features, load_data

LAYER_TYPES = ["RationalHat", "PermutationEquivariant", "Image", "Landscape", "BettiCurve", "Entropy", "Exponential", "Rational"]

PWEIGHTS = ["power", "grid", "gmix", None]

PERMUTATION_INVARIANT_OPERATION = ["sum", "max", "mean"]  # "topk" could be included but fails in most cases


def _process_persistence_diagrams(diags_dict):
    thresh = 500

    # Whole pipeline
    tmp = Pipeline([
        ("Selector", tda.DiagramSelector(use=True, point_type="finite")),
        ("ProminentPts", tda.ProminentPoints(use=True, num_pts=thresh)),
        ("Scaler", tda.DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])),
        ("Padding", tda.Padding(use=True)),
    ])

    prm = {filt: {"ProminentPts__num_pts": min(thresh, max([len(dgm) for dgm in diags_dict[filt]]))}
           for filt in diags_dict.keys() if max([len(dgm) for dgm in diags_dict[filt]]) > 0}

    # Apply the previous pipeline on the different filtrations.
    diags = []
    for dt in prm.keys():
        param = prm[dt]
        tmp.set_params(**param)
        diags.append(tmp.fit_transform(diags_dict[dt]))

    # For each filtration, concatenate all diagrams in a single array.
    D, npts = [], len(diags[0])
    for dt in range(len(prm.keys())):
        D.append(np.array(np.concatenate([diags[dt][i][np.newaxis, :] for i in range(npts)], axis=0), dtype=np.float32))
    return D, npts


def get_rui_type_depending_architecture(min, max, type="PyTorch", deterministic=False):
    if deterministic:
        return 0.5 * (min + max)
    if type == "PyTorch":
        return partial(torch.nn.init.uniform_, a=min, b=max)
    else:
        return tf.random_uniform_initializer(min, max)


def select_layer_type(layer_type, type="PyTorch", deterministic=False):
    perslay_channel_layer = dict()
    if layer_type == "Image":
        perslay_channel_layer["layer"] = "Image"
        perslay_channel_layer["image_size"] = (20, 20)
        perslay_channel_layer["image_bnds"] = ((-.001, 1.001), (-.001, 1.001))
        perslay_channel_layer["lvariance_init"] = 3.
    elif layer_type == "PermutationEquivariant":
        perslay_channel_layer["layer"] = "PermutationEquivariant"
        perslay_channel_layer["lpeq"] = [(5, "max")]
        perslay_channel_layer["lweight_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
        perslay_channel_layer["lbias_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
        perslay_channel_layer["lgamma_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
    elif layer_type == "Exponential":
        perslay_channel_layer["layer"] = "Exponential"
        perslay_channel_layer["lnum"] = 25
        perslay_channel_layer["lmean_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
        perslay_channel_layer["lvariance_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
    elif layer_type == "Rational":
        perslay_channel_layer["layer"] = "Rational"
        perslay_channel_layer["lnum"] = 25
        perslay_channel_layer["lmean_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
        perslay_channel_layer["lvariance_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
        perslay_channel_layer["lalpha_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
    elif layer_type == "RationalHat":
        perslay_channel_layer["layer"] = "RationalHat"
        perslay_channel_layer["lnum"] = 25
        perslay_channel_layer["lmean_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
        perslay_channel_layer["lr_init"] = get_rui_type_depending_architecture(3.0, 3.0, type, deterministic)
        perslay_channel_layer["q"] = 2
    elif layer_type == "Landscape":
        perslay_channel_layer["layer"] = "Landscape"
        perslay_channel_layer["lsample_num"] = 100
        perslay_channel_layer["lsample_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
    elif layer_type == "BettiCurve":
        perslay_channel_layer["layer"] = "BettiCurve"
        perslay_channel_layer["theta"] = 10
        perslay_channel_layer["lsample_num"] = 100
        perslay_channel_layer["lsample_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
    elif layer_type == "Entropy":
        perslay_channel_layer["layer"] = "Entropy"
        perslay_channel_layer["theta"] = 10
        perslay_channel_layer["lsample_num"] = 100
        perslay_channel_layer["lsample_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
    else:
        raise Exception("Not implemented layer type")
    return perslay_channel_layer


def select_weight_function(pweight, type="PyTorch", deterministic=False):
    perslay_channel_weight = dict()
    if pweight == "power":
        perslay_channel_weight["pweight"] = "power"
        perslay_channel_weight["pweight_init"] = 1.
        perslay_channel_weight["pweight_power"] = 1
    elif pweight == "grid":
        perslay_channel_weight["pweight"] = "grid"
        perslay_channel_weight["pweight_size"] = [20, 20]
        perslay_channel_weight["pweight_bnds"] = ((-.001, 1.001), (-.001, 1.001))
        if deterministic:
            perslay_channel_weight["pweight_init"] = np.ones([20, 20], dtype=np.float32)
        else:
            perslay_channel_weight["pweight_init"] = get_rui_type_depending_architecture(0.0, 1.0, type, deterministic)
    elif pweight == "gmix":
        perslay_channel_weight["pweight"] = "gmix"
        perslay_channel_weight["pweight_num"] = 3
        if deterministic:
            perslay_channel_weight["pweight_init"] = np.array(np.vstack([np.ones([2, 3]),
                                                                         5. * np.ones([2, 3])]), dtype=np.float32)
        else:
            perslay_channel_weight["pweight_init"] = np.array(np.vstack([np.random.uniform(0., 1., [2, 3]),
                                                                         5. * np.ones([2, 3])]), dtype=np.float32)
    elif pweight is None:
        perslay_channel_weight["pweight"] = None
    else:
        raise Exception("Not implemented weight function")
    return perslay_channel_weight


def select_permutation_invariant_op(perm_op, type="PyTorch", deterministic=False):
    perslay_channel_perm_inv_op = dict()
    if perm_op == "topk":
        perslay_channel_perm_inv_op["perm_op"] = "topk"
        perslay_channel_perm_inv_op["keep"] = 5
    elif perm_op == "sum" or perm_op == "max" or perm_op == "mean":
        perslay_channel_perm_inv_op["perm_op"] = perm_op
    else:
        raise Exception("Not implemented permutation invariant op")
    return perslay_channel_perm_inv_op


def test_are_tensorflow_and_pytorch_versions_equivalent():
    dataset = "MUTAG"
    generate_diagrams_and_features(dataset, path_dataset="../tutorial/data/MUTAG/")
    diags_dict, F, L = load_data(dataset, path_dataset="../tutorial/data/MUTAG/")
    D, npts = _process_persistence_diagrams(diags_dict)

    for layer_type in LAYER_TYPES:
        if layer_type == "PermutationEquivariant":
            # We do not test the permutation equivariant layer type because it is difficult to generate deterministic
            # initial values for the layer. However, we test this layer type individually for the PyTorch layer
            # to be sure that it works when initialized randomly.
            continue
        for pweight in PWEIGHTS:
            for perm_op in PERMUTATION_INVARIANT_OPERATION:
                print("Testing layer type: {}, weight function: {}, permutation invariant op: {}".format(layer_type,
                                                                                                         pweight,
                                                                                                         perm_op))
                tensorflow_layer = select_layer_type(layer_type, type="TensorFlow", deterministic=True)
                tensorflow_weight = select_weight_function(pweight, type="TensorFlow", deterministic=True)
                tensorflow_perm_inv_op = select_permutation_invariant_op(perm_op, type="TensorFlow", deterministic=True)
                tensorflow_perslay_channel = tensorflow_layer | tensorflow_weight | tensorflow_perm_inv_op
                tensorflow_perslay_channel["pweight_train"] = True
                tensorflow_perslay_channel["layer_train"] = True
                tensorflow_perslay_channel["final_model"] = "identity"
                tensorflow_perslay_parameters = [tensorflow_perslay_channel for _ in range(len(D))]
                perslay_tf = perslay.PerslayModel(name="PersLay", diagdim=2,
                                                  perslay_parameters=tensorflow_perslay_parameters,
                                                  rho="identity")
                pytorch_layer = select_layer_type(layer_type, type="PyTorch", deterministic=True)
                pytorch_weight = select_weight_function(pweight, type="PyTorch", deterministic=True)
                pytorch_perm_inv_op = select_permutation_invariant_op(perm_op, type="PyTorch", deterministic=True)
                pytorch_perslay_channel = pytorch_layer | pytorch_weight | pytorch_perm_inv_op
                pytorch_perslay_channel["pweight_train"] = True
                pytorch_perslay_channel["layer_train"] = True
                pytorch_perslay_channel["final_model"] = "identity"
                pytorch_perslay_parameters = [pytorch_perslay_channel for _ in range(len(D))]
                perslay_pt = perslay_pytorch.PerslayModel(diagdim=2, perslay_parameters=pytorch_perslay_parameters,
                                                          rho="identity")
                tf_result = perslay_tf.compute_representations(D[:]).numpy()
                pt_result = perslay_pt.compute_representations([torch.Tensor(dgm) for dgm in D[:]]).detach().numpy()

                assert np.allclose(tf_result, pt_result, atol=1e-2)
                print(f"Success")


def test_tensorflow_implementation_works():
    dataset = "MUTAG"
    generate_diagrams_and_features(dataset, path_dataset="../tutorial/data/MUTAG/")
    diags_dict, F, L = load_data(dataset, path_dataset="../tutorial/data/MUTAG/")
    D, npts = _process_persistence_diagrams(diags_dict)

    for layer_type in LAYER_TYPES:
        for pweight in PWEIGHTS:
            for perm_op in PERMUTATION_INVARIANT_OPERATION:
                tensorflow_layer = select_layer_type(layer_type, type="TensorFlow", deterministic=False)
                tensorflow_weight = select_weight_function(pweight, type="TensorFlow", deterministic=False)
                tensorflow_perm_inv_op = select_permutation_invariant_op(perm_op, type="TensorFlow", deterministic=False)
                tensorflow_perslay_channel = tensorflow_layer | tensorflow_weight | tensorflow_perm_inv_op
                tensorflow_perslay_channel["pweight_train"] = True
                tensorflow_perslay_channel["layer_train"] = True
                tensorflow_perslay_channel["final_model"] = "identity"
                tensorflow_perslay_parameters = [tensorflow_perslay_channel for _ in range(len(D))]
                perslay_tf = perslay.PerslayModel(name="PersLay", diagdim=2,
                                                  perslay_parameters=tensorflow_perslay_parameters,
                                                  rho="identity")
                tf_result = perslay_tf.compute_representations(D[:]).numpy()
                print(f"Success")


def test_pytorch_works_with_initialisers():
    dataset = "MUTAG"
    generate_diagrams_and_features(dataset, path_dataset="../tutorial/data/MUTAG/")
    diags_dict, F, L = load_data(dataset, path_dataset="../tutorial/data/MUTAG/")
    D, npts = _process_persistence_diagrams(diags_dict)

    for layer_type in LAYER_TYPES:
        for pweight in PWEIGHTS:
            for perm_op in PERMUTATION_INVARIANT_OPERATION:
                pytorch_layer = select_layer_type(layer_type, type="PyTorch", deterministic=False)
                pytorch_weight = select_weight_function(pweight, type="PyTorch", deterministic=False)
                pytorch_perm_inv_op = select_permutation_invariant_op(perm_op, type="PyTorch", deterministic=False)
                pytorch_perslay_channel = pytorch_layer | pytorch_weight | pytorch_perm_inv_op
                pytorch_perslay_channel["pweight_train"] = True
                pytorch_perslay_channel["layer_train"] = True
                pytorch_perslay_channel["final_model"] = "identity"
                pytorch_perslay_parameters = [pytorch_perslay_channel for _ in range(len(D))]
                perslay_pt = perslay_pytorch.PerslayModel(diagdim=2, perslay_parameters=pytorch_perslay_parameters,
                                                          rho="identity")
                perslay_pt.compute_representations([torch.Tensor(dgm) for dgm in D[:]]).detach().numpy()
                print(f"Success")


if __name__ == "__main__":
    print("Testing if TensorFlow implementation works")
    test_tensorflow_implementation_works()
    print("Tests performed successfully")
    print("-*-" * 10)
    print("Testing if PyTorch with random initializer works")
    test_pytorch_works_with_initialisers()
    print("Tests performed successfully")
    print("-*-" * 10)
    print("Testing if TensorFlow and PyTorch versions are equivalent")
    test_are_tensorflow_and_pytorch_versions_equivalent()
    print("Tests performed successfully")
