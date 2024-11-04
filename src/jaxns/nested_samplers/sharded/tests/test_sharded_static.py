from jaxns.nested_samplers.sharded.sharded_static import round_up_num_live_points


def test_round_up_num_live_points():
    assert round_up_num_live_points(10, 0.5, 1) == 10
    assert round_up_num_live_points(10, 0.5, 2) == 12
    assert round_up_num_live_points(10, 0.5, 3) == 12
