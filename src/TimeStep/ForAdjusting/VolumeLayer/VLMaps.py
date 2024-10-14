from copy import deepcopy

from .VLRings import VL_TO_RING


VOLUME_LAYER_TO_T_MAPS = [
    # Map 0
    {
        7: {2: 4, 4: 3, 6: 3, 8: 2, 10: 2, 12: 1, 14: 1},
        8: {2: 0, 4: 0, 6: 1, 8: 2},
        9: {2: 1, 4: 1, 6: 2, 8: 2, 10: 3, 12: 3, 14: 4},
        12: {6: 6, 8: 6, 10: 5, 12: 5},
        13: {2: 3, 4: 3, 6: 4, 8: 5},
        14: {2: 5, 4: 5, 6: 6, 8: 6},
        17: {2: 6},
    }
]


def get_volume_layer_to_t(map_index: int) -> tuple[dict, int]:
    """
    Get the volume layer to time-step mapping for a given map index.

    :param int map_index: The index of the map to get.
    :return: A tuple containing the volume layer to time-step mapping and the number of time-steps.
    """

    volume_layer_to_t = deepcopy(VOLUME_LAYER_TO_T_MAPS[map_index])

    max_t = max(max(layer_dict.values()) for layer_dict in volume_layer_to_t.values()) + 1

    for volume_id, layer_dict in VL_TO_RING.items():
        for layer_id in layer_dict.keys():
            if volume_id not in volume_layer_to_t:
                volume_layer_to_t[volume_id] = {layer_id: max_t}
            elif layer_id not in volume_layer_to_t[volume_id]:
                volume_layer_to_t[volume_id][layer_id] = max_t

    return volume_layer_to_t, max_t + 1


def get_t_to_volume_layers(vl_to_t: dict, num_t: int) -> dict:
    """
    Get a mapping from time-step to volume layers.

    :param dict vl_to_t: The volume layer to time-step mapping.
    :param int num_t: The number of time-steps.
    :return: A mapping from time-step to volume layers.
    """

    t_to_vls = {t: [] for t in range(num_t)}

    for volume_id, layer_dict in vl_to_t.items():
        for layer_id, t in layer_dict.items():
            t_to_vls[t].append((volume_id, layer_id))

    return t_to_vls
