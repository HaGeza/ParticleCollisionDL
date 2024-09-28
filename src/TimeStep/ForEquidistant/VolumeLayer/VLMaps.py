from copy import deepcopy


LAYER_MAPS = [
    # Map 0
    {
        7: {2: 4, 4: 3, 6: 2, 8: 2, 10: 2, 12: 1, 14: 1},
        8: {2: 0, 4: 0, 6: 1, 8: 3},
        9: {2: 1, 4: 1, 6: 2, 8: 2, 10: 2, 12: 3, 14: 4},
        12: {6: 6, 8: 6, 10: 5, 12: 5},
        13: {2: 2, 4: 3, 6: 4, 8: 5},
        14: {2: 5, 4: 5, 6: 6, 8: 6},
        17: {2: 6},
    }
]


def get_volume_layer_mapper(map_index: int) -> tuple[callable, int]:
    volume_layer_map = deepcopy(LAYER_MAPS[map_index])

    for volume_id, layer_dict in volume_layer_map.items():
        for layer_id in list(layer_dict.keys()):
            if layer_id % 2 == 0 and layer_id - 1 not in layer_dict:
                layer_dict[layer_id - 1] = layer_dict[layer_id]

    max_t = max(max(layer_dict.values()) for layer_dict in volume_layer_map.values()) + 1

    def map_vlt(row):
        volume_id = row["volume_id"]
        layer_id = row["layer_id"]
        if volume_id in volume_layer_map and layer_id in volume_layer_map[volume_id]:
            return volume_layer_map[volume_id][layer_id]
        else:
            return max_t

    return map_vlt, max_t + 1
