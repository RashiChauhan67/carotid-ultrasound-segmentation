import keras
import numpy as np

from carotid_seg.model import CoordAtt, build_model, segmentation_model


def test_parameter_counts_match_the_notebook():
    model = build_model()
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    # model.summary() in the notebook: 169,887 trainable + 1,766 non-trainable = 171,653.
    assert (model.count_params(), trainable, non_trainable) == (171_653, 169_887, 1_766)


def test_two_sigmoid_outputs_with_the_input_resolution():
    model = build_model(input_shape=(64, 64, 3))
    assert set(model.output_names) == {"segmentation", "edge_pred"}
    out = model.predict(np.random.rand(2, 64, 64, 3).astype("float32"), verbose=0)
    segmentation, edge = out["segmentation"], out["edge_pred"]
    assert segmentation.shape == edge.shape == (2, 64, 64, 1)
    assert 0.0 <= segmentation.min() and segmentation.max() <= 1.0


def test_segmentation_view_shares_the_layers_and_has_one_output():
    model = build_model(input_shape=(32, 32, 3))
    view = segmentation_model(model)
    assert len(view.outputs) == 1
    assert view.count_params() == model.count_params() == 171_653
    x = np.random.rand(2, 32, 32, 3).astype("float32")
    np.testing.assert_allclose(
        view.predict(x, verbose=0), model.predict(x, verbose=0)["segmentation"], atol=1e-5
    )


def test_coordinate_attention_keeps_the_shape_and_scales_by_at_most_one():
    layer = CoordAtt(32)
    x = np.random.rand(2, 16, 24, 32).astype("float32")
    out = np.asarray(layer(x))
    assert out.shape == x.shape
    assert (out <= x + 1e-6).all() and (out >= 0).all()  # attention weights are in (0, 1)


def test_save_and_load_round_trip(tmp_path):
    model = build_model(input_shape=(32, 32, 3))
    x = np.random.rand(2, 32, 32, 3).astype("float32")
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    before, after = model.predict(x, verbose=0), loaded.predict(x, verbose=0)
    for key in ("segmentation", "edge_pred"):
        np.testing.assert_allclose(before[key], after[key], atol=1e-5)
