from shapely.geometry import Polygon

from building_simplify.geometry import decode_polygon, make_source_tokens, make_target_tokens


def test_target_decode_repairs_rotation_induced_self_intersection():
    raw = Polygon([
        (379128.930747321, 3738555.2149466868),
        (379133.25167903095, 3738555.6370808105),
        (379133.31718421925, 3738554.9597429535),
        (379138.96403933864, 3738555.5092537194),
        (379139.81673689536, 3738546.792570454),
        (379138.2033483525, 3738546.6355668567),
        (379138.4981940529, 3738543.593091013),
        (379136.6158599208, 3738543.4062242457),
        (379137.2473523993, 3738536.965853511),
        (379130.77527211636, 3738536.338076943),
    ])
    target = Polygon([
        (379128.9682377096, 3738554.831269044),
        (379138.52809824684, 3738555.763149656),
        (379139.7041280386, 3738543.7128090565),
        (379136.6158599208, 3738543.4062242457),
        (379137.2473523993, 3738536.965853511),
        (379130.77527211636, 3738536.338076943),
    ])

    _, frame = make_source_tokens(raw, 5000)
    decoded = decode_polygon(make_target_tokens(target, frame), frame)

    assert decoded.geom_type == "Polygon"
    assert not decoded.is_empty
    assert decoded.is_valid
